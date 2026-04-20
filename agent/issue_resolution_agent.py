# 故障处理专家

import inspect

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY
from core.state import AgentState
from tools import db_tools, rag_tools
from utils.fliter_message import filter_messages_for_llm
from utils.monitor import audit_node
from utils.resilience import call_llm_with_fallback


# 在 issue_resolution_node 内部修改
def get_latest_user_input(messages):
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            # 处理多模态输入内容（可能是 list 也可能是 string）
            if isinstance(msg.content, list):
                return " ".join([p["text"] for p in msg.content if p["type"] == "text"])
            return msg.content
    return ""


@audit_node(model_name="gpt-4o-mini", input_price_per_m=1.05, output_price_per_m=4.2)
async def issue_resolution_node(state: AgentState, config=None):
    # 获取上下文数据
    messages = state.get("messages", [])
    user_symptom = state.get("user_symptom", "")
    order_id = state.get("order_id")
    vision_output = state.get("vision_output") or {}
    # 提取最后一条用户输入作为症状描述
    user_input = get_latest_user_input(messages)

    # 前置校验：如果没有订单号，先通过路由向用户索要
    if not order_id:
        return {
            "messages": [AIMessage(content="请提供您的订单号，以便我为您核实保修进度。")],
            "next": "supervisor",
            "reason": "缺少订单号，无法进入决策流程"
        }

    # 数据检索：获取订单与物流聚合信息
    order_info = await db_tools.get_combined_order_info(order_id)
    if not order_info:
        return {
            "messages": [AIMessage(content=f"抱歉，系统未找到订单号 {order_id}，请检查是否输入正确。")],
            "next": "supervisor"
        }

    # 优先使用签收日期(delivery_date)计算三包期，若未签收则用下单日期
    start_date = order_info["delivery_date"] or order_info["purchase_date"]
    # 提取视觉特征标签
    vision_tags = vision_output.get("visual_features", [])
    # 调用 RAG 工具
    decision_info = await rag_tools.get_policy_decision(
        user_text=user_input,
        vision_tags=vision_tags,
        purchase_date_str=start_date,
        category=vision_output.get("category"),
    )

    # 对消息进行过滤，剔除图片路径，只保留文字描述
    safe_messages = filter_messages_for_llm(messages)

    visual_features = vision_output.get('visual_features', ['无视觉证据'])

    system_prompt = inspect.cleandoc(f"""
        # Role
        你是一家售后系统决策客服。你负责基于 RAG 检索结果与订单数据，输出判罚方案，语气一定要平和，说话不能太死板。

        # 业务数据 (严格遵循)
        - 用户的请求：{user_symptom}
        - 购机/签收日期：{start_date}
        - 视觉特征: {visual_features}
        - 判罚结论：{decision_info['final_decision']}
        - 检索依据：{decision_info['policy_content']}
        
        Constraints
        1. 逻辑闭环：必须包含 [症状确认] -> [特征匹配] -> [政策依据] -> [最终判罚]四个要素。
        2. 格式严格：使用中文句号分隔，不要换行。
        3. 输出中要包含用户反馈了什么，你诊断出来了什么，最终的诊断结论是什么
        4. 输出中的判罚结论用中文给出
    """)

    prompts = [SystemMessage(content=system_prompt)] + safe_messages
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        base_url="https://api.chatanywhere.tech/v1",
    )
    # 调用模型
    response = await call_llm_with_fallback(llm, prompts, config=config)
    content = response.content

    return {
        "messages": [AIMessage(content=content)],
        "final_decision": decision_info["final_decision"],
        "next": "supervisor",  # 处理完当前问题，回到调度台
        "reason": f"已完成故障诊断与决策。判定结论：{decision_info['final_decision']}",
        "policy_context": decision_info["policy_content"],
    }
