import chromadb
from chromadb import EmbeddingFunction
from google import genai
from google.genai import types
from google.api_core import retry
import os

class GeminiEmbedding(EmbeddingFunction):
    def __init__(self, client):
        self.client = client
        self.document_mode = True

    @retry.Retry(predicate=lambda e: True)
    def __call__(self, docs):
        task = "retrieval_document" if self.document_mode else "retrieval_query"
        res = self.client.models.embed_content(
            model="models/text-embedding-004",
            contents=docs,
            config=types.EmbedContentConfig(task_type=task),
        )
        return [e.values for e in res.embeddings]

class VectorEngine:
    def __init__(self):
        API_KEY = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=API_KEY)

        self.embed_fn = GeminiEmbedding(self.client)

        self.chroma = chromadb.PersistentClient(path="./chroma_db")
        self.db = self.chroma.get_or_create_collection(
            name="google_rag",
            embedding_function=self.embed_fn
        )

    # Best practices for RAG - separate embedding fn for query and doc modes
    def embed_query(self, query):
        self.embed_fn.document_mode = False
        emb = self.embed_fn([query])[0]
        self.embed_fn.document_mode = True
        return emb

    def embed_documents(self, docs):
        return self.embed_fn(docs)

    # Document adding and searching operations
    def add_document(self, doc_id, text, metadata):
        self.db.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[metadata],
        )

    def search(self, query, n_results=3):
        result = self.db.query(query_texts=[query], n_results=n_results)
        if not result["documents"]:
            return []

        docs = result["documents"][0]
        metas = result["metadatas"][0]

        return [
            {"document": d, "metadata": m}
            for d, m in zip(docs, metas)
        ]

    # Count docs
    def count(self):
        return self.db.count()

    # LLM answer generator
    def generate_answer(self, query, docs):
        prompt = f"Answer the question using these documents:\n\nQUESTION: {query}\n"

        for i, d in enumerate(docs, 1):
            prompt += f"\nDOC {i}:\n{d[:1000]}\n"

        try:
            resp = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt
            )
            return resp.text
        except Exception as e:
            return f"Error generating answer: {e}"

