from pathlib import Path
import fitz
import re

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    # REMOVE author-heavy first lines (simple heuristic)
    text = re.sub(r"^.*ABSTRACT", "ABSTRACT", text, flags=re.DOTALL)

    # FIX broken hyphen line breaks
    text = re.sub(r"-\s+", "", text)

    return text.strip()

class PDFLoader:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)

    def load(self):
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}"
            )

        document = fitz.open(self.pdf_path)

        pages = []

        for page_num in range(len(document)):
            page = document[page_num]

            pages.append(
                {
                    "page_number": page_num + 1,
                    "text":clean_text(page.get_text())
                }
            )

        document.close()

        return pages