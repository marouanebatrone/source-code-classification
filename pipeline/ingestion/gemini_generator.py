import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from utils.hdfs_client import get_hdfs_client
from utils.logger import get_logger

load_dotenv()
log = get_logger("gemini_generator")

# --- Config ---
HDFS_DEST = "/data/ai"
LIMIT     = 100
TOPICS    = [
    "Blockchain with proof of work",
    "REST API with Flask for a book store",
    "Web scraper for e-commerce prices using BeautifulSoup",
    "Dijkstra shortest path algorithm",
    "Data cleaning pipeline using Pandas and NumPy",
    "Multi-threaded port scanner",
    "Linear Regression model using Scikit-learn",
    "AES-256 file encryption/decryption utility",
    "Asyncio-based chat server and client",
    "Automated AWS S3 backup script using Boto3",
    "N-Queens puzzle solver using recursion",
    "Stock price visualization with Plotly",
    "Neural Network from scratch with forward/backward pass",
    "SQL database migration script with SQLAlchemy",
    "Binary search tree with balancing logic",
]

def _clean_code(text: str) -> str:
    """Strip markdown fences from AI output."""
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()

def run(limit: int = LIMIT):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model       = genai.GenerativeModel("gemini-3-flash-preview")
    hdfs_client = get_hdfs_client()
    hdfs_client.makedirs(HDFS_DEST)

    generated = 10
    while generated < limit:
        topic  = TOPICS[generated % len(TOPICS)]
        prompt = (
            f"Write a complete, high-quality Python script for: {topic}. "
            "Output ONLY raw Python code. No explanations, no markdown."
        )
        log.info(f"[{generated + 1}/{limit}] Generating: {topic}")

        try:
            response = model.generate_content(prompt)
            code     = _clean_code(response.text)

            if not code:
                log.warning("Empty response, skipping.")
                continue

            hdfs_path = f"{HDFS_DEST}/ai_gemini_{generated}.py"
            with hdfs_client.write(hdfs_path, encoding="utf-8", overwrite=True) as w:
                w.write(code)

            generated += 1
            log.info(f"Saved: {hdfs_path}")
            time.sleep(5)

        except Exception as e:
            wait = 60 if "quota" in str(e).lower() else 10
            log.warning(f"Error: {e}. Waiting {wait}s...")
            time.sleep(wait)

    log.info(f"Done. {generated} AI scripts in HDFS.")


if __name__ == "__main__":
    run()