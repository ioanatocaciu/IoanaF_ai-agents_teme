import tema3
from tema3 import RAGSearchAgent, chunk_text, ingest_documents


class FakeCollection:
    def __init__(self):
        self.ids = []
        self.documents = []
        self.metadatas = []
        self.query_filter = None
        self.distances = [0.1]

    def get(self, where=None, include=None, limit=None):
        indexes = range(len(self.ids))
        if where:
            indexes = [
                index
                for index in indexes
                if all(self.metadatas[index].get(key) == value
                       for key, value in where.items())
            ]
        indexes = list(indexes)
        if limit is not None:
            indexes = indexes[:limit]
        return {
            "ids": [self.ids[index] for index in indexes],
            "metadatas": [self.metadatas[index] for index in indexes],
        }

    def delete(self, ids):
        keep = [index for index, item_id in enumerate(self.ids) if item_id not in ids]
        self.ids = [self.ids[index] for index in keep]
        self.documents = [self.documents[index] for index in keep]
        self.metadatas = [self.metadatas[index] for index in keep]

    def add(self, documents, metadatas, ids):
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def query(self, query_texts, n_results, where, include):
        self.query_filter = where
        indexes = [
            index for index, metadata in enumerate(self.metadatas)
            if metadata.get("source") == where["source"]
        ][:n_results]
        return {
            "documents": [[self.documents[index] for index in indexes]],
            "metadatas": [[self.metadatas[index] for index in indexes]],
            "distances": [[self.distances[index] for index in indexes]],
        }


def test_ingestion_records_source_and_page_for_each_chunk(tmp_path):
    source = tmp_path / "paragraphs.txt"
    source.write_text("A" * 25, encoding="utf-8")
    collection = FakeCollection()

    count = ingest_documents(collection, tmp_path)

    assert count == 1
    assert collection.metadatas == [{"source": "paragraphs.txt", "page": 1}]
    assert collection.documents == ["A" * 25]


def test_ingestion_replaces_existing_collection_content(tmp_path):
    (tmp_path / "new.txt").write_text("New content.", encoding="utf-8")
    collection = FakeCollection()
    collection.add(["old"], [{"source": "old.txt", "page": 1}], ["old-id"])

    ingest_documents(collection, tmp_path)

    assert collection.ids == ["new.txt:1:0"]
    assert collection.metadatas == [{"source": "new.txt", "page": 1}]


def test_answer_queries_only_selected_source_and_reports_page(monkeypatch):
    collection = FakeCollection()
    collection.add(
        ["Answer text"],
        [{"source": "paragraphs.txt", "page": 4}],
        ["chunk-1"],
    )
    monkeypatch.setattr(RAGSearchAgent, "_call_model", staticmethod(lambda messages: "Răspuns."))

    result = RAGSearchAgent(collection).answer("paragraphs.txt", "Întrebare?")

    assert collection.query_filter == {"source": "paragraphs.txt"}
    assert "paragraphs.txt, pagina 4" in result
    assert result.startswith("Răspuns.")


def test_answer_filters_chunks_above_cosine_distance_limit(monkeypatch):
    collection = FakeCollection()
    collection.add(
        ["Relevant text", "Distant text"],
        [
            {"source": "paragraphs.txt", "page": 1},
            {"source": "paragraphs.txt", "page": 2},
        ],
        ["near", "far"],
    )
    collection.distances = [0.3, 0.8]
    captured_messages = []
    monkeypatch.setattr(
        RAGSearchAgent,
        "_call_model",
        staticmethod(lambda messages: captured_messages.extend(messages) or "Răspuns."),
    )

    result = RAGSearchAgent(collection).answer("paragraphs.txt", "Întrebare?")

    assert "Relevant text" in captured_messages[1]["content"]
    assert "Distant text" not in captured_messages[1]["content"]
    assert "pagina 1" in result
    assert "pagina 2" not in result


def test_no_results_within_distance_limit_skips_llm(monkeypatch):
    collection = FakeCollection()
    collection.add(
        ["Distant text"],
        [{"source": "paragraphs.txt", "page": 2}],
        ["far"],
    )
    collection.distances = [0.8]
    monkeypatch.setattr(
        RAGSearchAgent,
        "_call_model",
        staticmethod(lambda messages: (_ for _ in ()).throw(AssertionError())),
    )

    result = RAGSearchAgent(collection).answer("paragraphs.txt", "Întrebare?")

    assert "Nu s-a găsit niciun rezultat" in result
    assert "0.5" in result


def test_unknown_source_is_reported_without_search(monkeypatch):
    collection = FakeCollection()
    collection.add(["Text"], [{"source": "known.txt", "page": 1}], ["chunk-1"])
    monkeypatch.setattr(collection, "query", lambda **kwargs: (_ for _ in ()).throw(AssertionError()))

    result = RAGSearchAgent(collection).answer("missing.txt", "Question?")

    assert "missing.txt" in result
    assert "known.txt" in result


def test_chunking_overlaps_chunks_and_rejects_invalid_overlap():
    assert chunk_text("abcdefgh", chunk_size=5, overlap=2) == ["abcde", "defgh"]

    try:
        chunk_text("text", chunk_size=5, overlap=5)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid overlap should raise ValueError.")


def test_collection_uses_cosine_distance(monkeypatch, tmp_path):
    monkeypatch.setattr("tema3.CHROMA_PATH", tmp_path / "chroma")

    collection = tema3.get_collection()

    assert collection.metadata["hnsw:space"] == "cosine"
