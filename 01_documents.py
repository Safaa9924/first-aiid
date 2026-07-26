"""
Stage 1 — Document Loading
==========================
Loads the source PDF (First Aid Reference Guide, 4th Edition — St. John
Ambulance Canada) using Docling, preserving structural elements
(headings, list items, captions) and saves the extracted raw text plus
the list of section headings for the next stage.
"""

import os
import json

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# ==================================================
# Configuration
# ==================================================

PDF_PATH = r"First_aid_reference_guide_V4.1_Public.pdf"

PUBLICATION_YEAR = 2019  # Fourth Edition, January 2019
SOURCE_TITLE = "First Aid Reference Guide, 4th Edition — St. John Ambulance Canada"

DATA_DIR = "data"

# ==================================================
# Initialize Docling Converter
# ==================================================

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.generate_picture_images = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


# ==================================================
# PDF Loader
# ==================================================

def load_pdf_document(pdf_path):
    """
    Load PDF with Docling while preserving document structure.

    Returns a dict with:
        source_file, raw_text, char_count, word_count, headings
    """

    result = converter.convert(pdf_path)
    doc = result.document

    text_parts = []
    headings = []

    for item, _ in doc.iterate_items():

        if not hasattr(item, "text"):
            continue

        text = item.text.strip()
        if not text:
            continue

        item_type = item.__class__.__name__

        if item_type == "SectionHeaderItem":
            text_parts.append(f"\n## {text}\n")
            headings.append(text)

        elif item_type == "ListItem":
            text_parts.append(f"• {text}")

        elif item_type == "CaptionItem":
            text_parts.append(f"\nCaption: {text}\n")

        else:
            text_parts.append(text)

    raw_text = "\n\n".join(text_parts)

    return {
        "source_file": pdf_path,
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "word_count": len(raw_text.split()),
        "headings": headings,
    }


def save_outputs(pdf_document, data_dir=DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "raw_text.txt"), "w", encoding="utf-8") as f:
        f.write(pdf_document["raw_text"])

    with open(os.path.join(data_dir, "headings.json"), "w", encoding="utf-8") as f:
        json.dump(pdf_document["headings"], f, ensure_ascii=False, indent=2)


# ==================================================
# Run Loader
# ==================================================

if __name__ == "__main__":

    pdf_document = load_pdf_document(PDF_PATH)
    save_outputs(pdf_document)

    print("=" * 60)
    print("PDF DOCUMENT SUMMARY")
    print("=" * 60)
    print("Source    :", SOURCE_TITLE)
    print("Characters:", pdf_document["char_count"])
    print("Words     :", pdf_document["word_count"])
    print("Headings  :", len(pdf_document["headings"]))

    print("\nFIRST 1000 CHARACTERS")
    print("=" * 60)
    print(pdf_document["raw_text"][:1000])
