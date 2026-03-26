from __future__ import annotations

from pathlib import Path

from app.rag.chunker import TokenChunker
from app.rag.vector_store import CodeDocument, FaissVectorStore
from app.utils.logger import get_logger


class CodeIngestor:
    """Ingest code files: read -> chunk(500) -> embedding -> FAISS."""

    def __init__(
        self,
        store: FaissVectorStore,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        logger_name: str = "codeinsight.rag.ingest",
    ) -> None:
        self.store = store
        self.chunker = TokenChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.logger = get_logger(logger_name)

    def ingest_directory(
        self,
        target_dir: str,
        include_suffixes: tuple[str, ...] = (".py", ".md", ".txt", ".js", ".ts", ".tsx", ".java"),
    ) -> dict[str, int]:
        base = Path(target_dir)
        if not base.exists():
            self.logger.warning("Ingest target not found: %s", target_dir)
            return {"files_read": 0, "chunks_created": 0, "chunks_indexed": 0}

        files_read = 0
        chunks_created = 0
        docs: list[CodeDocument] = []

        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if include_suffixes and path.suffix.lower() not in include_suffixes:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                self.logger.debug("Skip non-utf8 file: %s", path)
                continue

            files_read += 1
            chunks = self.chunker.split(file_path=str(path), text=content)
            chunks_created += len(chunks)
            for chunk in chunks:
                docs.append(
                    CodeDocument(
                        file_path=chunk.file_path,
                        content=chunk.content,
                        chunk_id=chunk.chunk_id,
                    )
                )

        chunks_indexed = self.store.add_documents(docs)
        self.logger.info(
            "Ingest finished: files_read=%d, chunks_created=%d, chunks_indexed=%d",
            files_read,
            chunks_created,
            chunks_indexed,
        )
        return {
            "files_read": files_read,
            "chunks_created": chunks_created,
            "chunks_indexed": chunks_indexed,
        }
