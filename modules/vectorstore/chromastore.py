# modules/vectorstore/chroma_store.py
import chromadb
from chromadb.config import Settings

class ChromaStore:
    def __init__(self, persist_dir: str):
        self.client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
        )
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids, embeddings, metadatas, documents):
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def query(self, embedding, n_results=5):
        return self.collection.query(query_embeddings=[embedding], n_results=n_results)
