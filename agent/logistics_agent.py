# 物流专家

import inspect

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY
from core.state import AgentState
from tools.db_tools import get_logistics_info
from utils.fliter_message import filter_messages_for_llm
from utils.monitor import audit_node
from utils.resilience import call_llm_with_fallback


@audit_node(model_name="gpt-4o-mini", input_price_per_m=1.05, output_price_per_m=4.2)
async def logistics_node(state: AgentState, config=None):
    # 检查是否有订单号
    order_id = state.get("order_id")
    # 如果没有订单号，模型需要询问用户
    if not order_id:
        # 这里可以直接让 LLM 生成询问语，或者直接返回
        return {
            "messages": [HumanMessage(content="为了帮您查询物流，请提供您的订单号。")],
            "next": "supervisor",  # 回到调度，等待用户输入
            "reason": "未提供订单号",
        }

    real_data = await get_logistics_info(str(order_id))

    # 对消息进行过滤，剔除图片路径，只保留文字描述
    safe_messages = filter_messages_for_llm(state["messages"])
    system_prompt = inspect.cleandoc(f"""
        # Role
        你是一位拥有10年经验的电商物流高级客服。你现在的目标是根据物流系统返回的【原始数据】，为用户提供一个准确、专业且有温度的【最终答复】。

        # Context (物流系统原始数据)
        {real_data}

        # User Input (用户的具体问题)
        {state["messages"][-1].content}

        # Goal (你的目标)
        1. 必须正面回答用户的问题。
        2. 将系统数据（如：2023-10-27）转化为更自然的表达（如：前天上午）。
        3. 如果用户问及“什么时候到”，请结合“estimate_delivery”给出合理预估。

        # Output Constraints (输出限制)
        - 严禁输出“根据系统显示”、“在提供的上下文中”等机械化词汇。
        - 严禁提及任何内部字段名或数据库术语。
        - 答案必须直接给用户看，不要包含任何前缀、后缀或自我介绍。
        - 语气要求：亲切、负责、高效。

        # Handling Scenarios
        - 如果数据查不到：告知用户可能由于信息延迟，建议2小时后重试。
        - 如果已签收：提醒用户检查快递柜或门口。
        - 如果运输中：告知当前具体位置，并给出安抚。
    """)
    # 构造消息列表 (System Prompt + 对话历史)
    prompts = [SystemMessage(content=system_prompt)] + safe_messages

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        base_url="https://api.chatanywhere.tech/v1",
    )
    # 调用模型
    response = await call_llm_with_fallback(llm, prompts, config=config)
    content = response.content
    # 去除 Markdown 的 JSON 标记
    content = content.replace("```json", "").replace("```", "").strip()

    return {
        "messages": [AIMessage(content=content)],
        "next": "supervisor",  # 处理完当前问题，回到调度台
        "reason": "",
    }
