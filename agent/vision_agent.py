# 视觉定损专家（多模态）

import base64
import inspect
import io
import json

from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from core.state import AgentState
from utils.resilience import call_llm_with_fallback


async def vision_node(state: AgentState, config=None):
    # ==========================================
    # 模式一：自动化测试模式 (Mock Bypass)
    # ==========================================
    # 从 state 中尝试获取测试注入的元数据
    test_metadata = state.get("test_metadata")

    if test_metadata:
        mock_output = test_metadata.get("mock_vision_output")
        test_id = test_metadata.get("test_id", "Unknown")

        # 场景 A：该测试用例有预设的视觉识别结果
        if mock_output:
            return {
                "vision_output": mock_output,  # 包含 category, visual_features, description
                "messages": [
                    AIMessage(content=f"视觉专家鉴定完成（模拟）：检测到 {', '.join(mock_output['visual_features'])}。")
                ],
                "next": "supervisor",
                "reason": f"测试模式：匹配案例 {test_id} 的预设识别结果"
            }

        # 场景 B：该测试用例预设为“不可见故障” (mock_vision_output 为 null)
        else:
            return {
                "vision_output": {
                    "visual_features": [],  # 返回空列表，触发 issue_agent 的纯文字 RAG
                    "category": None,
                    "description": "图片中未发现物理损伤，可能属于功能性/系统故障。"
                },
                "messages": [AIMessage(content="视觉鉴定完成：未发现外部可见损伤。")],
                "next": "supervisor",
                "reason": f"测试模式：案例 {test_id} 为功能性故障"
            }
    # ==========================================
    # 模式二：真实生产模式 (Real VLM Inference)
    # ==========================================

    # 寻找用户消息中的图片数据
    # 在 LangChain 中，多模态消息通常在 content 列表里
    image_data = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_data = part["image_url"]["url"]
                    break
        if image_data: break

    # 如果没找到图片，反问用户
    if not image_data:
        return {
            "messages": [AIMessage(content="请上传一张商品破损的照片，以便我为您进行损毁评估。")],
            "next": "supervisor",
            "reason": "等待用户上传图片"
        }

    # 图像预处理
    try:
        with Image.open(image_data) as img:
            img = img.convert("RGB")
            img.thumbnail((768, 768))  # Qwen2-VL 对这个尺寸识别效果很好
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=75)
            base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        return {"messages": [AIMessage(content=f"图片读取失败: {str(e)}")], "next": "supervisor",
                "reason": "图片读取失败，需要用户确认图像路径真实存在"}

    system_prompt = inspect.cleandoc("""
        # Role
        你是一位精通手机构造的视觉鉴定专家。
        
        # Task
        请分析用户上传的图片，并输出结构化的视觉故障特征。你的输出将用于匹配三包政策库。

        # 重点关注的特征标签示例
        - 屏幕类：放射状裂纹、碎屏、绿线、垂直条纹、黑斑、屏幕漏液、触控点失效。
        - 电池类：后盖隆起、中框缝隙、电池膨胀。
        - 接口类：尾插发绿、金属点发黑、接口变形。
        - 外观类：中框弯曲、后盖破碎、漆面剥落。

        # Output Format (必须是 JSON)
        {
          "visual_features": ["特征标签1", "特征标签2"],
          "category": 必须从以下类别中选一：[整机, 屏幕显示, 电池电源, 主板系统, 影像声学, 外观结构, 物流商务]。
          "description": "对毁坏情况的客观描述"
        }
    """)

    # 定义模型
    llm = ChatOllama(
        model="qwen3-vl:2b",
        base_url="http://localhost:11434",
        temperature=0,
    )

    # 构造多模态输入
    vision_input = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[
            {"type": "text", "text": "请分析这张图片。"},
            {"type": "image_url",
             "image_url": f"data:image/jpeg;base64,{base64_img}"}
        ])
    ]

    try:
        response = await call_llm_with_fallback(llm, vision_input, config=config)
        # 清洗并解析输出
        content = response.content.replace("```json", "").replace("```", "").strip()
        analysis_result = json.loads(content)

        # 4. 更新 state
        return {
            "vision_output": analysis_result,
            "messages": [
                AIMessage(content=f"视觉专家鉴定完成：检测到 {', '.join(analysis_result['visual_features'])}。")],
            "next": "supervisor",
            "reason": "定损评估及特征提取完成"
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content="抱歉，图片识别过程中遇到一点问题，能请您重新上传一张清晰的照片吗？")],
            "next": "supervisor",
            "reason": f"视觉识别故障: {str(e)}"
        }
