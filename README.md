# 基于多智能体的手机售后自动化决策系统

本项目是一个基于 **LangGraph** 编排的高性能、异步售后调度引擎。

## 🌟 核心亮点
- **多智能体协作**：Supervisor + Vision + Issue Resolution 三方协作，实现端到端售后闭环。
- **多模态判罚**：集成 Qwen-VL 实现视觉特征提取，自动比对 100 条三包政策库。
- **工业级架构**：FastAPI 接口服务化、Redis 会话持久化、异步非阻塞事件驱动。
- **FinOps 审计**：全链路 Token 消耗监测与成本核算系统。
- **自动化评估**：基于 RAGAS 框架，判定准确率达 90% 以上。

## 🛠️ 技术栈
- LangGraph / LangChain
- Redis Stack (Vector Store + Checkpointer)
- FastAPI / Uvicorn
- Docker Compose

## 🚀 快速启动
1. 配置 `.env` 文件填入 API Key。
2. 执行 `docker-compose up --build -d`。
