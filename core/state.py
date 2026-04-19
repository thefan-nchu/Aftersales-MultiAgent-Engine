# 定义 LangGraph 的 State

from operator import add
from typing import Annotated, TypedDict, List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    # 消息列表，Annotated[..., add_messages] 表示新消息会追加而不是覆盖
    messages: Annotated[List[AnyMessage], add_messages]
    # 当前处理的订单号
    order_id: Optional[str]
    # 视觉分析结果（如：屏幕碎裂程度 0.9）
    vision_output: Optional[dict]
    # RAG 检索到的政策片段
    policy_context: Optional[str]
    # 本次对话累计消耗的 Token 数
    token_usage: Annotated[int, add]
    # 下一个要执行的节点
    next: str
    # 各个 agent 对他们自己的输出的理由信息
    reason: Optional[str]
    # 标记当前是否处于退款流程中
    is_refunding: bool
    # 是否必须用户提供图片
    image_must: Optional[bool]
    # 记录最终决策，方便后续逻辑
    final_decision: Optional[str]
    # 用于存放测试用例中的 mock 数据
    test_metadata: Optional[dict]
    # Ragas 评估需要这个字段来判断 AI 是否在“胡编乱造”（即 Faithfulness 指标）
    policy_context: Optional[str]
    # 用户的提问
    user_symptom: str
    # 存放历史总结
    dialogue_summary: Optional[str]
    # 记录当前走了多少步，用于触发总结
    step_count: Annotated[int, add]
