# 构建 LangGraph 的拓扑结构
import os

from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import StateGraph, START, END
from langgraph.store.redis import AsyncRedisStore

from agent import *
from config import REDIS_URL
from core.state import AgentState


async def create_after_sales_graph():
    # 初始化状态图
    builder = StateGraph(AgentState)

    # 添加所有节点
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("logistics", logistics_node)
    builder.add_node("issue_resolution", issue_resolution_node)
    builder.add_node("vision", vision_node)
    builder.add_node("summarize", summarize_history_node)

    # 设置起始点
    builder.add_edge(START, "supervisor")

    # 设置条件路由 (Conditional Edges)
    # 根据 supervisor 修改的 state["next"] 值来决定去向
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "logistics_agent": "logistics",
            "issue_resolution_agent": "issue_resolution",
            "vision_agent": "vision",
            "summarize_agent": "summarize",
            "end": END
        }
    )

    # 设置循环边 (Loop back)
    # 每个专家处理完自己的任务后，把结果丢回给 supervisor 看看用户还有没有后续问题
    builder.add_edge("logistics", "supervisor")
    builder.add_edge("issue_resolution", "supervisor")
    builder.add_edge("vision", "supervisor")
    builder.add_edge("summarize", "supervisor")

    # 编译图
    # 加入 checkpointer来实现持久化记忆
    os.environ["REDIS_URL"] = REDIS_URL

    async with (
        AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer,
        AsyncRedisStore.from_conn_string(REDIS_URL) as store,
    ):
        await store.setup()
        await checkpointer.setup()

        graph = builder.compile(checkpointer=checkpointer, store=store)

    return graph
