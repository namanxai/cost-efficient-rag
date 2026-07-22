"""
Loaders for PDF / HTML / Markdown -> plain text.
"""
import os
from bs4 import BeautifulSoup
from pypdf import PdfReader


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator=" ")


def load_markdown_or_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext in (".html", ".htm"):
        return load_html(path)
    if ext in (".md", ".markdown", ".txt"):
        return load_markdown_or_text(path)
    raise ValueError(f"Unsupported file type: {ext}")
