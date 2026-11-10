
import re
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, chunk_size=800, chunk_overlap=120):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "Chunk overlap must be non-negative and smaller than chunk size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

    def clean_text(self, text):
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_pdf(self, pdf_path):
        path = Path(pdf_path)

        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        reader = PdfReader(str(path))

        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")

                if not decrypted:
                    raise ValueError(
                        "PDF is encrypted and requires a password."
                    )

            except Exception as error:
                raise ValueError(
                    "Unable to read encrypted PDF."
                ) from error

        documents = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            cleaned_text = self.clean_text(page_text)

            if cleaned_text:
                documents.append(
                    Document(
                        page_content=cleaned_text,
                        metadata={
                            "source": path.name,
                            "page": page_number
                        }
                    )
                )

        if not documents:
            raise ValueError(
                "No extractable text found. "
                "The PDF may contain scanned images."
            )

        return documents

    def split_documents(self, documents):
        if not documents:
            raise ValueError("No documents available for chunking.")

        chunks = self.text_splitter.split_documents(documents)

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = index

        return chunks

    def process_pdf(self, pdf_path):
        pages = self.extract_pdf(pdf_path)
        chunks = self.split_documents(pages)

        return {
            "pages": pages,
            "chunks": chunks,
            "total_pages": len(pages),
            "total_chunks": len(chunks)
        }
