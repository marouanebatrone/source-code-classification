import os
import time
import base64
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from utils.hdfs_client import get_hdfs_client
from utils.logger import get_logger

load_dotenv()
log = get_logger("github_scraper")

HDFS_DEST   = "/data/human"
LIMIT       = 100
QUERY       = "language:python created:<2022-01-01 stars:>50"
HEADERS     = {
    "Authorization": f"token {os.getenv('GITHUB_TOKEN')}",
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "AI-Detection-Project",
}

def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

def _get_tree(session, owner, name):
    """Try master then main branch."""
    for branch in ("master", "main"):
        url = f"https://api.github.com/repos/{owner}/{name}/git/trees/{branch}?recursive=1"
        r = session.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json().get("tree", [])
    return []

def _download_file(session, file_info) -> str | None:
    r = session.get(file_info["url"], headers=HEADERS, timeout=15)
    r.raise_for_status()
    raw = r.json().get("content", "")
    return base64.b64decode(raw).decode("utf-8") if raw else None

def run(limit: int = LIMIT):
    session     = _build_session()
    hdfs_client = get_hdfs_client()
    hdfs_client.makedirs(HDFS_DEST)

    collected = 0
    search_url = f"https://api.github.com/search/repositories?q={QUERY}&sort=updated"
    repos = session.get(search_url, headers=HEADERS, timeout=15).json().get("items", [])

    for repo in repos:
        if collected >= limit:
            break

        owner, name = repo["owner"]["login"], repo["name"]
        log.info(f"Exploring {owner}/{name}")

        tree = _get_tree(session, owner, name)
        py_files = [
            f for f in tree
            if f["path"].endswith(".py") and f.get("size", 0) < 100_000
        ][:5]

        for f_info in py_files:
            if collected >= limit:
                break
            try:
                code = _download_file(session, f_info)
                if not code:
                    continue

                clean = f_info["path"].replace("/", "_").replace("\\", "_")
                hdfs_path = f"{HDFS_DEST}/human_{owner}_{name}_{clean}"

                with hdfs_client.write(hdfs_path, encoding="utf-8", overwrite=True) as w:
                    w.write(code)

                collected += 1
                log.info(f"[{collected}/{limit}] Saved: {hdfs_path}")
                time.sleep(0.2)

            except Exception as e:
                log.warning(f"Skipping {f_info['path']}: {e}")

    log.info(f"Done. {collected} human scripts in HDFS.")


if __name__ == "__main__":
    run()