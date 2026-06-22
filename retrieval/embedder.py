"""
Pluggable embedding interface with two implementations:
  - BGEEmbedder: local bge-large-en-v1.5 via sentence-transformers
  - OpenAIEmbedder: OpenAI text-embedding-3-small via API

Key concept — asymmetric embedding:
  BGE was trained with different instructions for queries vs passages.
  Queries get a prefix ("Represent this sentence for retrieval: ") to
  tell the model "I'm searching for something." Passages get no prefix.
  Forgetting this drops retrieval quality by 5-10%. The embed_query()
  vs embed_texts() split enforces this distinction.
"""

from typing import Protocol


class Embedder(Protocol):
    """Any object with these three methods works as an embedder."""

    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed passage texts (no query prefix). Used for indexing."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query (with query prefix if needed)."""
        ...


class BGEEmbedder:
    """Local BGE-large-en-v1.5 via sentence-transformers.

    1024-dimensional vectors. Runs on CPU (~2s for 52 chunks).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_embedding_dimension()
        self._query_prefix = "Represent this sentence for retrieval: "

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        prefixed = self._query_prefix + text
        embedding = self._model.encode([prefixed], normalize_embeddings=True)
        return embedding[0].tolist()


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small via API.

    1536-dimensional vectors. Requires OPENAI_API_KEY env var.
    Essentially free for 52 chunks (~$0.0001).
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self._dimension = 1536

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=[text], model=self._model)
        return resp.data[0].embedding
