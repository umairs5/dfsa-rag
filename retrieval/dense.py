"""
Dense retrieval via Qdrant vector search.

Stores chunk embeddings in Qdrant (on-disk or in-memory) and retrieves
the top-k most similar chunks by cosine similarity.

Key concept — why prepend context_header:
  When embedding a chunk, we prepend its breadcrumb (e.g., "GEN > 3A
  Crypto Token Requirements > 3A.2 Prohibitions") to the text. This
  gives the embedding model structural context. Without it, two rules
  from different sections with similar wording embed too close together.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from retrieval.embedder import Embedder

COLLECTION = "dfsa_chunks"


class DenseRetriever:
    def __init__(
        self,
        embedder: Embedder,
        qdrant_path: str = "data/qdrant_store",
        qdrant_url: str | None = None,
    ):
        """
        Args:
            embedder: any Embedder (BGE or OpenAI)
            qdrant_url: if set, connect to a Qdrant server over HTTP
                        (e.g., "http://localhost:6333" for Docker setup)
            qdrant_path: ":memory:" for ephemeral, or a filesystem path
                         for on-disk persistence. Ignored if qdrant_url is set.
        """
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
        elif qdrant_path == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=qdrant_path)
        self.embedder = embedder

    def index(self, chunks: list[dict]) -> None:
        """Embed all chunks and upsert into Qdrant.

        Recreates the collection if it already exists.
        """
        texts = [
            f"{c.get('context_header', '')} | {c['text']}" for c in chunks
        ]
        embeddings = self.embedder.embed_texts(texts)

        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)

        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=self.embedder.dimension,
                distance=Distance.COSINE,
            ),
        )

        points = [
            PointStruct(
                id=i,
                vector=emb,
                payload={"chunk_id": chunks[i]["chunk_id"]},
            )
            for i, emb in enumerate(embeddings)
        ]
        self.client.upsert(collection_name=COLLECTION, points=points)

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """Return top-k (chunk_id, score) pairs by cosine similarity."""
        query_vec = self.embedder.embed_query(query)
        results = self.client.query_points(
            collection_name=COLLECTION,
            query=query_vec,
            limit=k,
        ).points
        return [(r.payload["chunk_id"], r.score) for r in results]
