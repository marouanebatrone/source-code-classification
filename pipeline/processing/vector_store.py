import os
import chromadb
from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
log = get_logger("vector_store")

_COLLECTION = "code_features"

def get_collection(reset: bool = False) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./ai_detection_db"))
    if reset:
        try:
            client.delete_collection(_COLLECTION)
            log.info("Collection reset.")
        except Exception:
            pass
    col = client.get_or_create_collection(_COLLECTION)
    log.info(f"Collection ready. Current count: {col.count()}")
    return col

def add_vectors(collection, ids: list, vectors: list, labels: list):
    metadatas = [
        {"label": int(l), "source": "ai" if l == 1 else "human"}
        for l in labels
    ]
    collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)