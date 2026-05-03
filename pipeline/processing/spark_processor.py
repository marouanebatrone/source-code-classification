import os
from dotenv import load_dotenv
from utils.hdfs_client import get_hdfs_client
from utils.logger import get_logger
from pipeline.features.extractor import extract
from pipeline.processing.vector_store import get_collection, add_vectors

load_dotenv()
log = get_logger("spark_processor")

SOURCES = [
    ("/data/human", 0),
    ("/data/ai",    1),
]

def _process_partition(path: str, label: int, collection):
    hdfs_client = get_hdfs_client()

    try:
        files = hdfs_client.list(path)
    except Exception as e:
        log.error(f"Cannot list {path}: {e}")
        return

    log.info(f"Found {len(files)} files in {path}")
    ids, vectors, labels = [], [], []

    for i, fname in enumerate(files):
        if not fname.endswith(".py"):
            continue
        try:
            with hdfs_client.read(f"{path}/{fname}", encoding="utf-8") as f:
                content = f.read()

            vec = extract(content)
            ids.append(f"id_{label}_{i}")
            vectors.append(vec)
            labels.append(label)

            if (i + 1) % 10 == 0:
                log.info(f"  Processed {i + 1}/{len(files)} from {path}")

        except Exception as e:
            log.warning(f"Skipping {fname}: {e}")

    if ids:
        add_vectors(collection, ids, vectors, labels)
        log.info(f"Stored {len(ids)} vectors from {path}")

def run():
    collection = get_collection(reset=True)
    for path, label in SOURCES:
        log.info(f"Processing {path} (label={label})...")
        _process_partition(path, label, collection)
    log.info(f"Done. Total vectors in DB: {collection.count()}")

if __name__ == "__main__":
    run()