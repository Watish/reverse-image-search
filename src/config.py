import os

VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "8192"))
INDEX_FILE_SIZE = int(os.getenv("INDEX_FILE_SIZE", "1024"))
METRIC_TYPE = os.getenv("METRIC_TYPE", "L2")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "default")
TOP_K = int(os.getenv("TOP_K", "10"))

UPLOAD_PATH = os.getenv("UPLOAD_PATH", "data/upload")
DATA_PATH = os.getenv("DATA_PATH", "data")

LOGS_NUM = int(os.getenv("logs_num", "0"))
