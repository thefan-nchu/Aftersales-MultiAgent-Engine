# evaluation_runner.py (核心数据采集逻辑)

import asyncio
import json
import os

from langchain_core.messages import HumanMessage, AIMessage

from config import REDIS_URL
from core.graph import create_after_sales_graph
from main import clear_test_sessions


async def run_batch_evaluation():
    await clear_test_sessions(redis_url=REDIS_URL)
    # 初始化图引擎
    app = await create_after_sales_graph()

    # 加载 50 条测试数据
    test_db_path = "../data/test_cases_db.json"
    if not os.path.exists(test_db_path):
        print(f"找不到测试集文件: {test_db_path}")
        return

    with open(test_db_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    eval_results = []
    print(f"开始采集评估数据，共 {len(test_cases)} 条...")

    for i, ticket in enumerate(test_cases[:1]):
        test_id = ticket["test_id"]
        print(f"[{i + 1}/{len(test_cases)}] 正在运行: {test_id}")

        # 构造输入
        initial_state = {
            "messages": [HumanMessage(content=ticket["user_input"])],
            "order_id": ticket["order_id"],
            "test_metadata": ticket  # 注入 mock 视觉数据
        }

        # 构造配置
        config = {"configurable": {"thread_id": f"eval_{test_id}"}}

        try:
            # 运行 Agent
            result = await app.ainvoke(initial_state, config=config)

            # 提取回复文本 (倒序查找非 JSON 的 AIMessage)
            answer = "未生成有效回复"
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    # 排除 supervisor 的 JSON 思考过程
                    if not (msg.content.strip().startswith("{") and "next_agent" in msg.content):
                        answer = msg.content
                        break

            # 构建 Ragas 要求的四元组
            eval_results.append({
                "question": ticket["user_input"],
                "answer": answer,
                # Ragas 要求 contexts 必须是 list of strings
                "contexts": [result.get("policy_context", "无参考背景")],
                "ground_truth": ticket["ground_truth"],
                # 用于业务判别准确率分析的辅助字段
                "system_decision": result.get("final_decision", "UNKNOWN"),
                "expected_decision": ticket["expected_decision"]
            })
        except Exception as e:
            print(f"案例 {test_id} 运行失败: {e}")

    # 保存采集到的原始结果
    os.makedirs("data", exist_ok=True)
    save_path = "../data/eval_dataset2.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)

    print(f"数据采集完成！结果已存至: {save_path}")


if __name__ == "__main__":
    asyncio.run(run_batch_evaluation())
