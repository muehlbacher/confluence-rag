"""M2 chunking invariants (no services needed):

- tables (and code) survive as single whole chunks
- every chunk begins with its breadcrumb
- no chunk exceeds max tokens except an intentionally-whole table/code block
- heading nesting is reflected in the breadcrumb section path
"""

from __future__ import annotations

from src.confluence.extract import ExtractedPage
from src.ingest.chunk import chunk_page
from src.ingest.parse import Block, parse_blocks


def make_page(breadcrumb="DOCS > Home > Guide") -> ExtractedPage:
    return ExtractedPage(
        page_id="42",
        space_key="DOCS",
        title="Guide",
        url="https://cf.example/display/DOCS/42",
        breadcrumb=breadcrumb,
    )


# Deterministic word-based counter so budgets are predictable in tests.
def word_count(text: str) -> int:
    return len(text.split())


# --- parse -------------------------------------------------------------------

def test_parse_blocks_classifies_and_drops_chrome():
    html = """
    <style>.x{color:red}</style>
    <h1>Title</h1>
    <p>First para.</p>
    <h2>Sub</h2>
    <ul><li>one</li><li>two</li></ul>
    <table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
    <pre>def f():\n    return 1</pre>
    <script>evil()</script>
    """
    blocks = parse_blocks(html)
    kinds = [(b.kind, b.level) for b in blocks]
    assert ("heading", 1) in kinds
    assert ("heading", 2) in kinds
    assert any(b.kind == "table" for b in blocks)
    assert any(b.kind == "code" for b in blocks)
    # No macro CSS/JS leaked into any block.
    joined = " ".join(b.text for b in blocks)
    assert "color:red" not in joined and "evil()" not in joined


def test_parse_table_rows_rendered_line_per_row():
    html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    (table,) = parse_blocks(html)
    assert table.kind == "table"
    assert table.text == "A | B\n1 | 2"


# --- chunk invariants --------------------------------------------------------

def test_every_chunk_begins_with_its_breadcrumb():
    page = make_page()
    blocks = [
        Block("heading", "Setup", level=1),
        Block("text", "Install the thing."),
        Block("heading", "Details", level=2),
        Block("text", "More detail here."),
    ]
    chunks = chunk_page(page, blocks, max_tokens=100, count_tokens=word_count)
    assert chunks
    for ch in chunks:
        assert ch.text.startswith(ch.breadcrumb)
        assert ch.breadcrumb.startswith(page.breadcrumb)


def test_heading_nesting_in_breadcrumb():
    page = make_page()
    blocks = [
        Block("heading", "Setup", level=1),
        Block("heading", "Networking", level=2),
        Block("text", "VLAN config."),
    ]
    (chunk,) = chunk_page(page, blocks, max_tokens=100, count_tokens=word_count)
    assert chunk.breadcrumb == "DOCS > Home > Guide > Setup > Networking"


def test_sibling_heading_pops_stack():
    page = make_page()
    blocks = [
        Block("heading", "A", level=1),
        Block("heading", "A1", level=2),
        Block("text", "under a1"),
        Block("heading", "B", level=1),  # pops A1 and A
        Block("text", "under b"),
    ]
    chunks = chunk_page(page, blocks, max_tokens=100, count_tokens=word_count)
    a1 = next(c for c in chunks if "under a1" in c.text)
    b = next(c for c in chunks if "under b" in c.text)
    assert a1.breadcrumb == "DOCS > Home > Guide > A > A1"
    assert b.breadcrumb == "DOCS > Home > Guide > B"


def test_table_survives_as_single_whole_chunk():
    page = make_page()
    big_table = "\n".join(f"r{i} | v{i}" for i in range(200))  # far over budget
    blocks = [
        Block("heading", "Data", level=1),
        Block("table", big_table),
    ]
    chunks = chunk_page(page, blocks, max_tokens=50, count_tokens=word_count)
    table_chunks = [c for c in chunks if c.kind == "table"]
    assert len(table_chunks) == 1
    tc = table_chunks[0]
    assert tc.whole is True
    assert big_table in tc.text  # not split


def test_no_text_chunk_exceeds_max_tokens():
    page = make_page()
    long_para = " ".join(f"word{i}" for i in range(500))
    blocks = [Block("heading", "Big", level=1), Block("text", long_para)]
    max_tokens = 60
    chunks = chunk_page(page, blocks, max_tokens=max_tokens, count_tokens=word_count)
    assert len(chunks) > 1  # it was split
    for ch in chunks:
        if ch.whole:
            continue  # tables/code may exceed; there are none here anyway
        assert ch.token_count <= max_tokens, (ch.token_count, ch.text[:60])


def test_code_block_kept_whole_and_over_budget_allowed():
    page = make_page()
    code = "\n".join(f"line_{i} = {i}" for i in range(100))
    blocks = [Block("heading", "Code", level=1), Block("code", code)]
    chunks = chunk_page(page, blocks, max_tokens=20, count_tokens=word_count)
    (code_chunk,) = [c for c in chunks if c.kind == "code"]
    assert code_chunk.whole is True
    assert code in code_chunk.text


def test_oversized_table_splits_on_rows_with_repeated_header():
    page = make_page()
    header = "Name | Recht"
    rows = [f"user{i} | role{i}" for i in range(400)]
    table = "\n".join([header] + rows)
    blocks = [Block("heading", "Rechte", level=1), Block("table", table)]
    # Small hard cap forces splitting; whole-block ceiling is hard_max_tokens.
    chunks = chunk_page(
        page, blocks, max_tokens=50, hard_max_tokens=40, count_tokens=word_count
    )
    table_chunks = [c for c in chunks if c.kind == "table"]
    assert len(table_chunks) > 1  # it was split
    for tc in table_chunks:
        assert tc.whole is False
        assert tc.token_count <= 40  # respects the hard ceiling
        # header row repeated on every piece for context
        assert header in tc.text
    # every data row is present across the pieces
    joined = "\n".join(tc.text for tc in table_chunks)
    assert "user399 | role399" in joined and "user0 | role0" in joined


def test_code_block_splits_on_lines_when_over_hard_cap():
    page = make_page()
    code = "\n".join(f"line_{i} = compute({i})" for i in range(300))
    blocks = [Block("heading", "Code", level=1), Block("code", code)]
    chunks = chunk_page(
        page, blocks, max_tokens=30, hard_max_tokens=25, count_tokens=word_count
    )
    code_chunks = [c for c in chunks if c.kind == "code"]
    assert len(code_chunks) > 1
    for cc in code_chunks:
        assert cc.token_count <= 25


def test_chunk_ids_are_sequential_per_page():
    page = make_page()
    blocks = [
        Block("heading", "H", level=1),
        Block("text", "a"),
        Block("table", "x | y"),
        Block("text", "b"),
    ]
    chunks = chunk_page(page, blocks, max_tokens=100, count_tokens=word_count)
    assert [c.chunk_id for c in chunks] == [f"42:{i}" for i in range(len(chunks))]
