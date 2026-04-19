FROM python:3.11-slim

WORKDIR /app

# 1. 修改系统源
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# 2. 升级 pip
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. ！！！终极方案：强行安装核心包（跳过依赖计算） ！！！
# 我们先装 pydantic 和 jiter 的最新稳定版，不让它在旧版本里打转
RUN pip install --no-cache-dir jiter pydantic-core pydantic -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 强行安装 LangChain 家族（同样避开解析卡顿）
RUN pip install --no-cache-dir langchain-core langchain-openai langchain-community langgraph -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 最后安装剩下的所有依赖，此时绝大部分依赖已满足，秒过
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
