from langchain_core.messages import HumanMessage


def filter_messages_for_llm(messages):
    """
    将消息列表中的多模态内容（图片）替换为文本占位符，
    防止非视觉模型在处理长 Base64 或本地路径时崩溃。
    """
    clean_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
            # 提取文字内容
            text_content = " ".join([part["text"] for part in msg.content if part["type"] == "text"])
            # 检查是否有图片
            if any(part["type"] == "image_url" for part in msg.content):
                text_content += " [用户已上传附件图片]"
            clean_messages.append(HumanMessage(content=text_content))
        else:
            # SystemMessage 或 AIMessage 直接透传
            clean_messages.append(msg)
    return clean_messages
