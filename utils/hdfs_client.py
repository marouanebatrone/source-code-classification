import os
from hdfs import InsecureClient
from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
log = get_logger("hdfs")

def get_hdfs_client() -> InsecureClient:
    url  = os.getenv("HDFS_URL")
    user = os.getenv("HDFS_USER")
    client = InsecureClient(url, user=user)
    log.info(f"HDFS client ready → {url}")
    return client