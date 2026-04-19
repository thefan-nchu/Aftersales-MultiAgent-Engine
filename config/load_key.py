import os


def load_key():
    os.environ["DASHSCOPE_API_KEY"] = "sk-**"
    os.environ["REDIS_PASSWORD"] = "**"
    os.environ["TONGYI_API_KEY"] = "sk-**"
    os.environ["OPENAI_API_KEY"] = "sk-**"
    os.environ["REDIS_URL"] = "redis://:**@localhost:6379/0"
