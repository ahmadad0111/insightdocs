"""Load a PDF into page-level records with light cleaning."""
from pathlib import Path
import re

import fitz  # PyMuPDF


def clean_text(text: str) -> str:
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # join words split by a hyphen at a line break: "trans- former" -> "transformer"
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return text.strip()


class PDFLoader:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)

    def load(self):
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        document = fitz.open(self.pdf_path)
        pages = []
        for page_num in range(len(document)):
            raw = document[page_num].get_text()
            cleaned = clean_text(raw)
            if cleaned:
                pages.append({"page_number": page_num + 1, "text": cleaned})
        document.close()
        return pages
