"""Retrieval eval harness: hit-rate@k and MRR over eval/dataset.jsonl.

Reports both reranker-off (RRF fusion order) and reranker-on so the reranker's
contribution is measurable on the same dataset. Ranking is page-level: chunks
are deduped to their page in rank order, and a question is a hit if any of its
gold page ids appears in the top-k pages.

For the reranker-on ranking we reorder *all* retrieved candidates by
cross-encoder score (no score threshold) — the threshold is the M4 no-context
gate, not a ranking cutoff. run with:

    python -m eval.run_eval
    python -m eval.run_eval --k 5 --top-k 20
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

from config import get_settings
from src.retrieval.rerank import Reranker
from src.retrieval.search import Candidate, Searcher

_DATASET = os.path.join(os.path.dirname(__file__), "dataset.jsonl")


def load_dataset(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pages_in_order(chunk_page_ids: Sequence[str]) -> List[str]:
    """Dedupe page ids to first-occurrence order."""
    seen: Dict[str, None] = {}
    for pid in chunk_page_ids:
        if pid not in seen:
            seen[pid] = None
    return list(seen.keys())


def first_gold_rank(ranked_pages: List[str], gold: Sequence[str]) -> int:
    """1-based rank of the first gold page, or 0 if absent."""
    gold_set = set(gold)
    for i, pid in enumerate(ranked_pages, start=1):
        if pid in gold_set:
            return i
    return 0


def metrics(ranks_pages: List[List[str]], golds: List[Sequence[str]], k: int):
    hits, rr = 0, 0.0
    n = len(golds)
    for pages, gold in zip(ranks_pages, golds):
        rank = first_gold_rank(pages, gold)
        if rank and rank <= k:
            hits += 1
        if rank:
            rr += 1.0 / rank
    return hits / n, rr / n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="hit-rate@k (default 5)")
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="candidates retrieved per query (default RETRIEVE_TOP_K)",
    )
    parser.add_argument("--dataset", default=_DATASET)
    args = parser.parse_args()

    settings = get_settings()
    dataset = load_dataset(args.dataset)
    searcher = Searcher(settings)
    reranker = Reranker(settings)

    golds = [row["gold_page_ids"] for row in dataset]
    baseline_pages: List[List[str]] = []
    reranked_pages: List[List[str]] = []

    for row in dataset:
        cands: List[Candidate] = searcher.search(row["question"], top_k=args.top_k)
        baseline_pages.append(pages_in_order([c.page_id for c in cands]))
        # Pure reordering: no threshold, keep all candidates.
        ranked = reranker.rerank(row["question"], cands, top_n=len(cands), score_min=0.0)
        reranked_pages.append(pages_in_order([r.page_id for r in ranked]))

    k = args.k
    base_hit, base_mrr = metrics(baseline_pages, golds, k)
    rr_hit, rr_mrr = metrics(reranked_pages, golds, k)

    print(f"dataset: {len(dataset)} questions | top_k={args.top_k or settings.retrieve_top_k}")
    print(f"{'config':<16}{'hit-rate@' + str(k):<16}{'MRR':<10}")
    print(f"{'reranker OFF':<16}{base_hit:<16.3f}{base_mrr:<10.3f}")
    print(f"{'reranker ON':<16}{rr_hit:<16.3f}{rr_mrr:<10.3f}")

    # Show misses (reranker-on) to aid dataset/tuning review.
    misses = [
        dataset[i]["question"]
        for i in range(len(dataset))
        if first_gold_rank(reranked_pages[i], golds[i]) == 0
        or first_gold_rank(reranked_pages[i], golds[i]) > k
    ]
    if misses:
        print(f"\nreranker-on misses (not in top-{k}):")
        for q in misses:
            print(f"  - {q}")

    searcher.close()
    reranker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
