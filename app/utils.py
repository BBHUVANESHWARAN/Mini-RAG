import io
from typing import List, Tuple
from pypdf import PdfReader

ALLOWED = {".pdf", ".txt", ".md"}


def is_allowed(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in ALLOWED)


def extract_text_from_pdf(file_bytes: bytes) -> List[Tuple[int, str]]:
    # returns list of (page_number, text)
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, text))
    return pages


def extract_text_from_txt(file_bytes: bytes) -> List[Tuple[int, str]]:
    text = file_bytes.decode(errors="ignore")
    return [(1, text)]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    # simple char-based chunker
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = max(end - overlap, end)
    return chunks