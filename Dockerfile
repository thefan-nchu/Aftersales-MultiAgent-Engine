# 使用轻量级 Python 镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置镜像源
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装系统级依赖（处理图片可能需要）
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有代码
COPY . .

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 启动命令
# 注意：host 必须设为 0.0.0.0 才能在容器外访问
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]