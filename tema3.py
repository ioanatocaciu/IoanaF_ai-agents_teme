import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_SOURCES = PROJECT_ROOT / "data_sources"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RESULT_COUNT = 3
MAX_COSINE_DISTANCE = 0.5
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def read_document_pages(path: Path) -> list[tuple[int, str]]:
    """Return each document page as a (page number, text) pair."""
    if path.suffix.casefold() == ".pdf":
        from pypdf import PdfReader

        return [
            (page_number, page.extract_text() or "")
            for page_number, page in enumerate(PdfReader(path).pages, start=1)
        ]
    return [(1, path.read_text(encoding="utf-8"))]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than it.")
    text = text.strip()
    if not text:
        return []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break
    return chunks


def ingest_documents(collection: Any, data_directory: Path = DATA_SOURCES) -> int:
    """Index all UTF-8 text and PDF files under data_directory."""
    files = sorted(path for path in data_directory.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files found in {data_directory}.")

    sources = [path.name for path in files]
    if len(sources) != len(set(sources)):
        raise ValueError(
            "Files in data_sources must have unique filenames because source metadata "
            "stores the filename."
        )

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    for path in files:
        source = path.name
        for page_number, page_text in read_document_pages(path):
            for chunk_number, chunk in enumerate(chunk_text(page_text)):
                documents.append(chunk)
                metadatas.append({"source": source, "page": page_number})
                ids.append(f"{source}:{page_number}:{chunk_number}")

    if not documents:
        raise ValueError(f"No extractable text found in {data_directory}.")

    previous = collection.get(include=["metadatas"])
    previous_ids = previous.get("ids", [])
    if previous_ids:
        collection.delete(ids=previous_ids)

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return len(documents)


def get_collection() -> Any:
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    if (collection.metadata or {}).get("hnsw:space") != "cosine":
        client.delete_collection(name=COLLECTION_NAME)
        collection = client.create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return collection


class RAGSearchAgent:
    def __init__(self, collection: Any) -> None:
        self.collection = collection

    @staticmethod
    def _call_model(messages: list[dict[str, str]]) -> str:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
        model = os.getenv("OPEN_ROUTER_MODEL_NAME")
        if not api_key or not model:
            raise RuntimeError(
                "Set OPEN_ROUTER_API_KEY and OPEN_ROUTER_MODEL_NAME in .env "
                "to run the LLM agent."
            )

        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        response = client.chat.completions.create(model=model, messages=messages)
        return (response.choices[0].message.content or "").strip()

    def answer(self, source: str, question: str) -> str:
        matching_chunks = self.collection.get(
            where={"source": source}, include=["metadatas"], limit=1
        )
        if not matching_chunks.get("ids"):
            available = self.collection.get(include=["metadatas"])
            sources = sorted(
                {
                    metadata["source"]
                    for metadata in available.get("metadatas", [])
                    if metadata and "source" in metadata
                }
            )
            if not sources:
                return "Nu există date indexate."
            return f"Nu există date pentru {source!r}. Fișiere disponibile: {', '.join(sources)}."

        results = self.collection.query(
            query_texts=[question],
            n_results=RESULT_COUNT,
            where={"source": source},
            include=["documents", "metadatas", "distances"],
        )
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        if not documents:
            return f"Nu am găsit fragmente relevante în {source!r}."

        context_parts = []
        citations = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if distance > MAX_COSINE_DISTANCE:
                continue
            page_number = metadata.get("page", "necunoscută")
            context_parts.append(
                f"[Fișier: {metadata.get('source', source)}, pagina {page_number}]\n{document}"
            )
            citations.append((metadata.get("source", source), page_number))

        if not context_parts:
            return (
                f"Nu s-a găsit niciun rezultat în {source!r} cu distanța cosine "
                f"maximă de {MAX_COSINE_DISTANCE}."
            )

        answer = self._call_model(
            [
                {
                    "role": "system",
                    "content": (
                        "Răspunde în limba întrebării și folosește numai contextul oferit. "
                        "Dacă nu conține răspunsul, spune clar că nu l-ai găsit. "
                        "Citează numele fișierului și pagina pentru afirmațiile folosite."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Întrebare: {question}\n\n"
                        f"Context:\n{chr(10).join(context_parts)}"
                    ),
                },
            ]
        )
        citations_text = ", ".join(
            f"{filename}, pagina {page}" for filename, page in dict.fromkeys(citations)
        )
        return f"{answer}\n\nSurse: {citations_text}"


def main() -> None:
    collection = get_collection()
    chunk_count = ingest_documents(collection)
    print(f"Au fost indexate {chunk_count} fragmente din {DATA_SOURCES}.")

    source = input("Numele fișierului (de exemplu paragraphs.txt): ").strip()
    question = input("Întrebarea: ").strip()
    if not source or not question:
        raise ValueError("Numele fișierului și întrebarea nu pot fi goale.")
    print(RAGSearchAgent(collection).answer(source, question))


if __name__ == "__main__":
    main()
