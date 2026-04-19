import os


def load_key():
    # 设置阿里云 DashScope 的 Key
    os.environ["DASHSCOPE_API_KEY"] = "sk-4500e056fb41401da4a1f3660639bb67"
    os.environ["REDIS_PASSWORD"] = "wodemima"
    os.environ["TONGYI_API_KEY"] = "sk-4500e056fb41401da4a1f3660639bb67"
    os.environ["OPENAI_API_KEY"] = "sk-xxQdZhffEkSiVxlHXM2RR5PeC1cD9yB9bgVsJkr6Hsh31Pll"
    os.environ["REDIS_URL"] = "redis://:wodemima@localhost:6379/0"
