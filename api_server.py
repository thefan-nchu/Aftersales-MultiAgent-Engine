import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from config import REDIS_URL
from core.graph import create_after_sales_graph


# Pydantic 数据模型定义
class TicketRequest(BaseModel):
    """客户端请求模型"""
    order_id: str = Field(..., description="订单编号", json_schema_extra={"example": "10001"})
    user_input: str = Field(..., description="用户描述的故障或诉求",
                            json_schema_extra={"example": "刚收到手机屏幕碎了"})
    image_path: Optional[str] = Field(None, description="本地图片路径（用于视觉定损）")
    thread_id: Optional[str] = Field(None, description="会话ID，用于保持多轮对话记忆")


class AgentResponse(BaseModel):
    """服务端返回模型"""
    thread_id: str
    status: str = "success"
    decision: str = Field(..., description="判罚结论：REFUND_FULL, PAID_REPAIR 等")
    reply: str = Field(..., description="给用户的人性化回复文本")
    reason: str = Field(..., description="系统内部的决策理由")


# 生命周期管理
# 用于存放全局单例对象
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 【启动阶段】 ---
    print("🚀 [System] 正在初始化售后 Agent 引擎...")
    try:
        # 预加载图引擎，确保 Redis 连接和索引初始化完成
        app_state["agent_engine"] = await create_after_sales_graph()
        print("✅ [System] Agent 引擎加载成功，接口已就绪")
    except Exception as e:
        print(f"❌ [Error] 引擎初始化失败: {e}")
        raise e

    yield  # --- 运行中 ---

    # --- 【关闭阶段】 ---
    print("🛑 [System] 正在关闭服务，清理资源...")
    app_state.clear()


# ==========================================
# FastAPI 实例初始化
# ==========================================

app = FastAPI(
    title="手机售后智能决策引擎 API",
    description="基于 LangGraph + Redis + RAG 的多智能体自动化售后处理系统",
    version="2.0.0",
    lifespan=lifespan
)


# ==========================================
# 核心业务接口
# ==========================================

@app.post("/v1/aftersales/process", response_model=AgentResponse)
async def process_aftersales_ticket(req: TicketRequest):
    """
    处理售后请求的核心接口
    """
    agent_app = app_state.get("agent_engine")
    if not agent_app:
        raise HTTPException(status_code=503, detail="Agent engine is not initialized")

    # 处理会话 ID：如果客户端没传，则生成一个新的 UUID
    current_thread_id = req.thread_id or f"api_session_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": current_thread_id}}

    # 构造多模态输入内容
    # 遵循 LangChain 标准格式：[{"type": "text", "text": "..."}, {"type": "image_url", ...}]
    message_content = [{"type": "text", "text": req.user_input}]
    if req.image_path:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": req.image_path}
        })

    # 执行 Agent 决策流
    try:
        # 初始状态注入
        initial_state = {
            "messages": [HumanMessage(content=message_content)],
            "order_id": req.order_id
        }

        # 调用 ainvoke（这会自动触发 monitor 的审计和 Redis 的持久化）
        result = await agent_app.ainvoke(initial_state, config=config)

        # 设置过期时间为 24 小时
        r = Redis.from_url(REDIS_URL)
        thread_id = current_thread_id
        prefixes = ["checkpoint", "checkpoint_write", "checkpoint_latest"]
        for prefix in prefixes:
            await r.expire(f"{prefix}:{thread_id}", 86400)

        # 解析结果
        final_messages = result.get("messages", [])
        ai_reply = "系统处理完成，请查看详情"

        # 倒序查找最后一条真实的 AI 文本答复
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and msg.content:
                # 排除中间的 JSON 过程
                if not (msg.content.strip().startswith("{") and "next_agent" in msg.content):
                    ai_reply = msg.content
                    break

        return AgentResponse(
            thread_id=current_thread_id,
            decision=result.get("final_decision", "UNKNOWN"),
            reply=ai_reply,
            reason=result.get("reason", "无详细理由")
        )

    except Exception as e:
        print(f"[API Error] 处理工单时发生崩溃: {e}")
        raise HTTPException(status_code=500, detail=f"内部决策链路故障: {str(e)}")


# ==========================================
# 系统状态与日志接口
# ==========================================

@app.get("/v1/system/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "online",
        "engine_ready": "agent_engine" in app_state,
        "environment": os.environ.get("ENV", "development")
    }


@app.get("/v1/audit/summary")
async def get_cost_summary(limit: int = 10):
    """读取最近的审计日志摘要"""
    log_path = "data/audit_log.jsonl"
    if not os.path.exists(log_path):
        return {"message": "No logs found"}

    logs = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f.readlines()[-limit:]:
            logs.append(json.loads(line))
    return logs


if __name__ == "__main__":
    # 运行 API 服务器
    # 访问 http://127.0.0.1:8000/docs 查看可视化文档
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
