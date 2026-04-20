# Token 统计与成本计算
import json
import os
import time
from contextvars import ContextVar
from datetime import datetime

import aiofiles
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage

audit_context = ContextVar("audit_context", default=None)


# 提取逻辑
def extract_usage(response):
    input_t, output_t = 0, 0
    try:
        if getattr(response, "llm_output", None):
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}
            input_t = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
            output_t = usage.get("completion_tokens") or usage.get("output_tokens", 0)
    except:
        pass

    # 针对 ChatTongyi 的深度适配
    if input_t == 0 and hasattr(response, "generations") and response.generations:
        try:
            gen_info = response.generations[0][0].generation_info
            usage = gen_info.get("token_usage") or {}
            input_t = usage.get("input_tokens", 0)
            output_t = usage.get("output_tokens", 0)
        except:
            pass

    print(f'response, input_t, output_t: {response}, {input_t}, {output_t}')

    return input_t, output_t


# Handler 内部存储数据
class UnifiedPricingHandler(BaseCallbackHandler):
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        in_t, out_t = extract_usage(response)
        # 直接存入实例属性，不使用 ContextVar
        self.input_tokens = in_t
        self.output_tokens = out_t


# 直接从 Handler 获取数据
def audit_node(model_name, input_price_per_m, output_price_per_m):
    def decorator(func):
        async def wrapper(state=None, config=None, *args, **kwargs):
            # 创建本次调用专属的 Handler 实例
            handler = UnifiedPricingHandler()
            audit_context.set(None)  # 每一轮开始前清空

            new_config = config.copy() if config else {}
            new_config["callbacks"] = [handler]

            start_time = time.time()
            # 执行业务逻辑
            result = await func(state, config=new_config, *args, **kwargs)
            latency = time.time() - start_time

            # 提取数据
            manual_data = audit_context.get()
            if manual_data:
                in_t = manual_data.get("input", 0)
                out_t = manual_data.get("output", 0)
            else:
                in_t = handler.input_tokens
                out_t = handler.output_tokens

            if in_t > 0 or out_t > 0:
                cost = (in_t * input_price_per_m / 1_000_000 + out_t * output_price_per_m / 1_000_000)

                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "node": func.__name__,
                    "model": model_name,
                    "input_t": in_t,
                    "output_t": out_t,
                    "total_t": in_t + out_t,
                    "cost": round(cost, 8),
                    "latency": round(latency, 2),
                }

                # 使用异步写文件
                os.makedirs("../data", exist_ok=True)
                async with aiofiles.open("../data/audit_log.jsonl", "a", encoding="utf-8") as f:
                    await f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                # 清理
                audit_context.set(None)
                # print(f"[AUDIT] {func.__name__} 记账成功: {in_t + out_t} tokens")

            # 清洗消息元数据保持 State 纯净
            if result and "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    result["messages"][-1] = AIMessage(content=last_msg.content)

            return result

        return wrapper

    return decorator
