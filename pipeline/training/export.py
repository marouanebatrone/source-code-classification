import os
import csv
import chromadb
from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
log = get_logger("export")

CSV_PATH = "extracted_features.csv"
HEADER   = ["label", "perplexity", "comment_ratio", "id_length", "ast_nodes", "ast_depth"]

def run():
    col = chromadb.PersistentClient(
        path=os.getenv("CHROMA_PATH", "./ai_detection_db")
    ).get_collection("code_features")

    data = col.get(include=["embeddings", "metadatas"])
    total = len(data["ids"])
    log.info(f"Exporting {total} records to {CSV_PATH}...")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for i in range(total):
            label = data["metadatas"][i]["label"]
            vec   = data["embeddings"][i]
            writer.writerow([label, *vec])

    log.info(f"Done. Saved to {CSV_PATH}")

if __name__ == "__main__":
    run()