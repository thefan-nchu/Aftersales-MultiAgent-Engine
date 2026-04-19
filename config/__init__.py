import os

from .load_key import load_key

load_key()
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TONGYI_API_KEY = os.environ.get("TONGYI_API_KEY")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")

REDIS_URL = os.environ.get(
    "REDIS_URL",
    f"redis://:{REDIS_PASSWORD}@localhost:6379/0"
)
