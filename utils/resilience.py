# 双模型故障切换回退
from langchain_ollama import ChatOllama


async def call_llm_with_fallback(llm_primary, prompts, config=None):
    """
    故障切换逻辑
    """
    try:
        # 尝试使用主模型
        return await llm_primary.ainvoke(prompts, config=config)
    except Exception as e:
        print(f"主模型异常: {e}, 正在切换备用模型...")

        # 实例化备用模型 (如 Qwen-Plus)
        fallback_llm = ChatOllama(
            model="deepseek-r1:1.5b",
            base_url="http://localhost:11434"
        )
        return await fallback_llm.ainvoke(prompts, config=config)
