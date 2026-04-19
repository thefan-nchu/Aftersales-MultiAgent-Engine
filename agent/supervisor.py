# 路由分发智能体

import inspect
import json

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import TONGYI_API_KEY
from core.state import AgentState
from utils.fliter_message import filter_messages_for_llm
from utils.monitor import audit_node
from utils.resilience import call_llm_with_fallback
from utils.satety import check_prompt_injection, mask_pii_data


def get_text_content(content) -> str:
    """
    安全提取纯文本内容
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # 从多模态列表中提取所有类型为 text 的片段并拼接
        texts = [item["text"] for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return " ".join(texts)

    return ""


@audit_node(model_name="qwen-plus", input_price_per_m=2, output_price_per_m=12)
async def supervisor_node(state: AgentState, config=None):
    # 检查步数或消息长度
    if len(state["messages"]) > 20:
        return {"next": "summarize_node"}

    # 逻辑：
    # 1. 分析 state["messages"]
    # 2. 判断意图：['logistics', 'refund', 'vision', 'end']
    # 3. 返回更新后的 state

    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and msg.content:
            user_symptom = msg.content
            break

    # 对消息进行过滤，剔除图片路径，只保留文字描述
    safe_messages = filter_messages_for_llm(state["messages"])

    # 安全拦截：检查提示词注入
    user_msg = get_text_content(state["messages"][-1])

    # print(state)
    if check_prompt_injection(user_msg):
        return {
            "messages": [AIMessage(content="系统检测到异常指令请求，出于安全考虑，本次操作已被终止。")],
            "next": "end",
            "reason": "检测到恶意 Prompt Injection 攻击"
        }

    # 隐私脱敏
    for msg in safe_messages:
        if isinstance(msg.content, str):
            msg.content = mask_pii_data(msg.content)

    system_prompt = inspect.cleandoc("""
        # Role
        售后系统路由中枢 (Event-Driven Router)
        
        # Expertise & Boundary
        你负责根据对话状态决定控制权归属。你必须严格遵循【事件驱动】原则：
        - logistics_agent: 仅限查询【轨迹、位置、预计时间】（解决“在哪儿”的问题）。
        - issue_resolution_agent: 负责处理一切关于“商品本身”及“订单实物”的异常诉求。包括：产品故障诊断（软硬件）、损毁定损、核对三包时效。
        - vision_agent: 拥有视觉分析能力，负责基于图像证据的损毁评定。
        - policy_agent: 拥有规则库检索权限，负责合规性判定与政策解答。
        - end: 结束当前回合。
                
        # Execution Rules (优先级最高)
        1. **视觉前置准则（重要）**：
        - 只要用户的诉求涉及【屏幕（绿线、裂纹）、外壳（变形、划痕）、接口（损坏、发绿）】等外观部件。
        - **必须首先指派 vision_agent 进行特征提取**，严禁直接跳过视觉节点进入决策节点。
        
        2. **停止准则 (Wait for User)**:
        - 检查消息历史的【最后一条】。
        - 如果最后一条消息是 `AIMessage`（由专家发出），说明系统正在等待用户反馈。
        - **此时你必须返回 `next_agent: "end"`**。严禁连续两次指派同一个专家，严禁在用户开口前替专家补话。
        
        3. **触发准则 (User Trigger)**:
        - 只有当最后一条消息是 `HumanMessage`（由用户发出）时，你才被激活并指派专家。
        - 你根据用户的最新诉求和当前状态（order_id是否具备、vision_output是否具备）来选择最合适的资源。
        
        4. **单次指派原则**:
        - 每一轮用户的输入，只能驱动【一次】专家的指派。一旦专家完成了它的逻辑（无论是给答案还是提问），该轮流程即刻终止。
        
        5. **识别故障属性**:
        - 物理性/外观损伤 (Visible Damage) -> 必须有图，指派 vision_agent。
        - 内部性能/软件故障 (Functionality Issue) -> 无需图片，直接指派 issue_resolution_agent。  
        
        # State Logic
        - `order_id`: 仅从 HumanMessage 中提取，严禁脑补。
        
        # Output Format (JSON)
        请只返回一个唯一的 JSON 对象。严禁重复输出，严禁输出多个 JSON 块。
        只返回以下 JSON，不要有任何 Markdown 代码块标签或解释：
        {
          "next_agent": "xxx",
          "reason": "简述决策：是由于专家已回复进入等待，还是用户发起新诉求进行指派",
          "order_id": "提取到的订单号，如无则为 null",
          "image_must": "如果是故障处理请求，识别出是否必须用户提供图片，识别结果为 true/false。否则为 null"
        }
    """)
    # 构造消息列表 (System Prompt + 对话历史)
    prompts = [SystemMessage(content=system_prompt)] + safe_messages

    # 定义模型
    # llm = ChatOllama(
    #     model="deepseek-r1:1.5b",
    #     base_url="http://localhost:11434",
    # )
    # llm = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     # model="deepseek-r1:1.5b",
    #     # api_key="ollama",
    #     openai_api_key="sk-xxQdZhffEkSiVxlHXM2RR5PeC1cD9yB9bgVsJkr6Hsh31Pll",
    #     base_url="https://api.chatanywhere.tech/v1",
    #     # base_url="http://localhost:11434/v1" # one-api的访问地址
    # )
    llm = ChatTongyi(
        model="qwen-plus",
        api_key=TONGYI_API_KEY,
    )
    # 调用模型
    response = await call_llm_with_fallback(llm, prompts, config=config)

    content = response.content

    # 去除 Markdown 的 JSON 标记
    content = content.replace("```json", "").replace("```", "").strip()
    # 解析并更新状态
    try:
        decision = json.loads(content)
        next_node = decision.get("next_agent", "end")

        if next_node == "end":
            return {
                "next": "end",
                "order_id": state.get("order_id"),
                "reason": decision.get("reason"),
            }
        else:
            return {
                "next": next_node,
                "order_id": decision.get("order_id") if decision.get("order_id")
                else state.get("order_id"),
                "reason": decision.get("reason"),
                "image_must": decision.get("image_must") if decision.get("image_must")
                else state.get("image_must"),
                "user_symptom": user_symptom
            }
    except Exception as e:
        print(f"解析故障: {e}, 原始输出: {content}")
        return {"next": "end"}
