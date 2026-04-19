# 敏感数据脱敏
import re


def mask_pii_data(text: str) -> str:
    """
    使用正则对敏感数据进行脱敏
    """
    if not text or not isinstance(text, str):
        return text

    # 1. 脱敏手机号 (11位数字)
    text = re.sub(r'1[3-9]\d{9}', '[HIDDEN_PHONE]', text)

    # 2. 脱敏邮箱
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[HIDDEN_EMAIL]', text)

    # 3. 脱敏详细地址 (简单示例：匹配xx省xx市xx区)
    text = re.sub(r'(.+省|.+自治区)(.+市)(.+区|.+县)', r'\1\2[HIDDEN_ADDRESS]', text)

    return text


def check_prompt_injection(text: str) -> bool:
    """
    检查是否存在提示词注入风险
    """
    print(text)
    danger_keywords = ["忽略之前的指令", "ignore previous instructions", "system administrator", "管理员模式"]
    return any(kw in text.lower() for kw in danger_keywords)
