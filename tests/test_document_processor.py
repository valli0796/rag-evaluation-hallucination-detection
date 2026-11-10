
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.document_processor import DocumentProcessor


def test_processor_initialization():
    processor = DocumentProcessor(
        chunk_size=500,
        chunk_overlap=100
    )

    assert processor.chunk_size == 500
    assert processor.chunk_overlap == 100


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        DocumentProcessor(chunk_size=0)


def test_invalid_chunk_overlap():
    with pytest.raises(ValueError):
        DocumentProcessor(
            chunk_size=500,
            chunk_overlap=500
        )


def test_clean_text():
    processor = DocumentProcessor()

    text = "RAG   combines   retrieval.\n\n\nIt uses documents."

    cleaned = processor.clean_text(text)

    assert cleaned == "RAG combines retrieval.\n\nIt uses documents."


def test_missing_pdf():
    processor = DocumentProcessor()

    with pytest.raises(FileNotFoundError):
        processor.extract_pdf("missing_file.pdf")


def test_unsupported_file():
    processor = DocumentProcessor()

    with pytest.raises(ValueError):
        processor.extract_pdf("document.txt")


def test_extract_sample_pdf():
    processor = DocumentProcessor()

    pdf_path = Path("data/documents/sample.pdf")

    assert pdf_path.exists()

    pages = processor.extract_pdf(pdf_path)

    assert len(pages) > 0
    assert all(page.page_content.strip() for page in pages)
    assert all("page" in page.metadata for page in pages)


def test_split_documents():
    processor = DocumentProcessor(
        chunk_size=100,
        chunk_overlap=20
    )

    documents = [
        Document(
            page_content=(
                "RAG combines information retrieval "
                "with language models. " * 10
            ),
            metadata={
                "source": "test.pdf",
                "page": 1
            }
        )
    ]

    chunks = processor.split_documents(documents)

    assert len(chunks) > 0
    assert all("chunk_id" in chunk.metadata for chunk in chunks)


def test_empty_documents():
    processor = DocumentProcessor()

    with pytest.raises(ValueError):
        processor.split_documents([])
