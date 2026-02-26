"""
Document text extraction service using Docling.

Extracts clean text and structured tables from PDF and DOCX files.
Used by the /document-text-extraction API endpoint.
"""

from collections import defaultdict
from pathlib import Path
from typing import Optional

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption, DocumentConverter


# ---------------------------------------------------------------------------
# Docling document converter (PDF/DOCX)
# ---------------------------------------------------------------------------
PDF_PIPELINE_OPTIONS = PdfPipelineOptions(
    do_ocr=True,
    do_layout=True,
    do_structure=True,
    do_table_structure=True,
    do_cell_matching=True,
)

DOC_CONVERTER = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PDF_PIPELINE_OPTIONS,
        )
    }
)


def _collect_table_texts(doc_dict: dict) -> set[str]:
    """Collect raw text content that appears inside table cells."""
    table_texts: set[str] = set()
    for table in doc_dict.get("tables", []):
        for cell in table.get("data", {}).get("table_cells", []):
            text = (cell.get("text") or "").strip()
            if text:
                table_texts.add(text)
    return table_texts


def _extract_clean_text(doc_dict: dict, table_texts: set[str]) -> str:
    """Extract clean linear text while avoiding duplication of table content."""
    texts_map = {t["self_ref"]: t for t in doc_dict.get("texts", [])}
    groups_map = {g["self_ref"]: g for g in doc_dict.get("groups", [])}
    tables_map = {f"#/tables/{i}": t for i, t in enumerate(doc_dict.get("tables", []))}

    visited: set[str] = set()
    output: list[str] = []

    def get_child_ref(child: dict) -> Optional[str]:
        return child.get("$ref")

    def resolve_ref(ref: str) -> None:
        if ref in visited:
            return
        visited.add(ref)

        # Text blocks
        if ref in texts_map:
            block = texts_map[ref]
            if block.get("content_layer") == "furniture":
                return

            text = (block.get("text") or "").strip()
            if not text:
                return

            label = block.get("label", "")

            # Skip text that is part of a table (we will represent tables separately)
            if label != "section_header" and text in table_texts:
                return

            if label == "section_header":
                output.append(f"\n\n## {text}\n")
            else:
                output.append(text)

        # Grouped content
        elif ref in groups_map:
            group = groups_map[ref]
            for child in group.get("children", []):
                child_ref = get_child_ref(child)
                if child_ref:
                    resolve_ref(child_ref)

        # Tables: skip for linear text, but keep traversal alive
        elif ref in tables_map:
            return

    # Traverse main body
    for child in doc_dict.get("body", {}).get("children", []):
        ref = child.get("$ref")
        if ref:
            resolve_ref(ref)

    # Include orphan groups not linked from body
    all_groups = doc_dict.get("groups", [])
    for i in range(len(all_groups)):
        group_ref = f"#/groups/{i}"
        if group_ref not in visited and group_ref in groups_map:
            resolve_ref(group_ref)

    return "\n".join(output)


def _extract_structured_tables(doc_dict: dict) -> list[dict]:
    """Extract tables as structured JSON with headers and rows."""
    all_structured_tables: list[dict] = []

    for table_index, table in enumerate(doc_dict.get("tables", [])):
        cells = table.get("data", {}).get("table_cells", [])
        if not cells:
            continue

        headers: dict[int, str] = {}
        rows: dict[int, dict[int, str]] = defaultdict(dict)

        for cell in cells:
            row_idx = cell.get("start_row_offset_idx")
            col_idx = cell.get("start_col_offset_idx")
            text = (cell.get("text") or "").strip()
            if text == "" or row_idx is None or col_idx is None:
                continue

            if cell.get("column_header"):
                headers[col_idx] = text
            else:
                rows[row_idx][col_idx] = text

        if not headers:
            continue

        sorted_header_indices = sorted(headers.keys())
        sorted_headers = [headers[idx] for idx in sorted_header_indices]
        structured_table: list[dict[str, str]] = []

        for row_idx in sorted(rows.keys()):
            row_data = rows[row_idx]
            structured_row: dict[str, str] = {}
            for col_idx, header in zip(sorted_header_indices, sorted_headers):
                structured_row[header] = row_data.get(col_idx, "")
            structured_table.append(structured_row)

        all_structured_tables.append(
            {
                "table_index": table_index,
                "headers": sorted_headers,
                "rows": structured_table,
            }
        )

    return all_structured_tables


def extract_document_text_and_tables(file_path: Path) -> tuple[str, list[dict]]:
    """
    Run Docling on a document and return clean text plus structured tables.

    Args:
        file_path: Path to a PDF or DOCX file.

    Returns:
        Tuple of (clean_text, tables) where tables is a list of
        {"table_index", "headers", "rows"} dicts.
    """
    conv_result = DOC_CONVERTER.convert(str(file_path))
    doc = conv_result.document
    doc_dict = doc.export_to_dict()

    table_texts = _collect_table_texts(doc_dict)
    clean_text = _extract_clean_text(doc_dict, table_texts)
    tables = _extract_structured_tables(doc_dict)
    return clean_text, tables
