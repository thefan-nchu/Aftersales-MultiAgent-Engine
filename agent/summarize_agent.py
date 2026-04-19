from langchain_core.messages import SystemMessage, RemoveMessage
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY
from utils.fliter_message import filter_messages_for_llm
from utils.monitor import audit_node
from utils.resilience import call_llm_with_fallback


@audit_node(model_name="gpt-4o-mini", input_price_per_m=1.05, output_price_per_m=4.2)
async def summarize_history_node(messages, config=None):
    # 留最后两条不总结，保持即时语境
    safe_messages = filter_messages_for_llm(messages[:-2])
    summary_prompt = "请简要总结以下对话的核心诉求和当前进度，剔除寒暄和重复信息。字数控制在100字内。"
    prompts = [SystemMessage(content=summary_prompt)] + safe_messages

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        base_url="https://api.chatanywhere.tech/v1",
    )
    # 调用模型
    response = await call_llm_with_fallback(llm, prompts, config=config)

    return {
        "dialogue_summary": response.content,
        # 使用 RemoveMessage 物理删除 Redis 中旧的消息，防止消息列表无限增长
        "messages": [RemoveMessage(id=m.id) for m in messages[:-2]],
        "reason": "执行了历史记录压缩总结"
    }
