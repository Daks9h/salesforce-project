"""
ChromaDB RAG Engine Module
Handles the ingestion of FAQ data into a vector database (ChromaDB)
and provides semantic search capabilities using Google Generative AI embeddings
or fast, reliable local embeddings.
"""
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from chromadb.utils import embedding_functions
import json
import os
import hashlib
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FastLocalEmbeddingFunction(EmbeddingFunction):
    """
    Fast, reliable, zero-download embedding function using feature hashing
    and L2-normalized bag-of-words vectors. Runs instantly in any environment
    without requiring external model downloads or internet access.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            vec = np.zeros(self.dim, dtype=np.float32)
            # Tokenize into clean words
            words = text.lower().replace("\n", " ").split()
            for word in words:
                clean_word = "".join(c for c in word if c.isalnum())
                if not clean_word:
                    continue
                # Hash word to dimension index
                h = int(hashlib.md5(clean_word.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
                vec[idx] += sign

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec.tolist())
        return embeddings

class RAGEngine:
    """
    Core RAG (Retrieval-Augmented Generation) Engine.
    Manages vector embeddings and similarity search for FAQs.
    """
    def __init__(self, db_path=None, api_key=None):
        """
        Initialize the ChromaDB client and embedding function.
        
        Args:
            db_path (str, optional): Path to store the persistent ChromaDB files.
            api_key (str, optional): Google API key for Gemini embeddings.
        """
        if db_path is None:
            db_path = "/tmp/chroma_db" if os.environ.get("VERCEL") else "./chroma_db"
            
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        
        if api_key:
            try:
                self.embedding_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                    api_key=api_key
                )
            except Exception as e:
                logger.warning(f"Failed to initialize GoogleGenerativeAiEmbeddingFunction ({e}), using fast local embeddings.")
                self.embedding_fn = FastLocalEmbeddingFunction()
        else:
            self.embedding_fn = FastLocalEmbeddingFunction()
            
        self.collection = self.client.get_or_create_collection(
            name="it_support_kb",
            embedding_function=self.embedding_fn
        )

        # Auto-ingest if collection is empty (e.g. serverless cold-start)
        try:
            if self.collection.count() == 0:
                kb_path = Path(__file__).parent / "knowledge_base.json"
                if kb_path.exists():
                    self.ingest_json(kb_path)
        except Exception as e:
            logger.warning(f"Auto-ingest notice: {e}")

    def ingest_json(self, json_path):
        """Read knowledge_base.json and add to vector db"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load knowledge base: {e}")
            return 0
            
        documents = []
        metadatas = []
        ids = []
        
        idx = 0
        for category in data.get("categories", []):
            cat_name = category.get("name", "General")
            for faq in category.get("faqs", []):
                question = faq.get("question", "")
                answer = faq.get("answer", "")
                
                # Store question + answer for contextual retrieval
                text_to_embed = f"Category: {cat_name}\nQuestion: {question}\nAnswer: {answer}"
                
                documents.append(text_to_embed)
                metadatas.append({
                    "category": cat_name,
                    "question": question,
                    "answer": answer
                })
                ids.append(f"faq_{idx}")
                idx += 1
                
        if documents:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            return len(documents)
        return 0

    def query(self, user_query, n_results=3):
        """
        Search for relevant context using semantic similarity.
        
        Args:
            user_query (str): The search query from the user.
            n_results (int): Number of top results to return.
            
        Returns:
            list: A list of dictionaries containing matched FAQs and their scores.
        """
        try:
            results = self.collection.query(
                query_texts=[user_query],
                n_results=n_results
            )
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            return []
        
        formatted_results = []
        if results and results.get('metadatas') and results['metadatas'][0]:
            for i in range(len(results['metadatas'][0])):
                meta = results['metadatas'][0][i]
                distance = results['distances'][0][i] if ('distances' in results and results['distances']) else 0
                formatted_results.append({
                    "question": meta['question'],
                    "answer": meta['answer'],
                    "category": meta['category'],
                    "score": max(0.0, 1.0 - (distance / 2.0))
                })
        return formatted_results

if __name__ == "__main__":
    engine = RAGEngine()
    kb_path = Path(__file__).parent / "knowledge_base.json"
    count = engine.ingest_json(kb_path)
    logger.info(f"Ingested {count} documents.")
    
    test_query = "How do I pair my wireless headphones?"
    logger.info(f"\nQuery: {test_query}")
    matches = engine.query(test_query)
    for m in matches:
        logger.info(f"- [{m['category']}] {m['question']} (Score: {m['score']:.2f})")
