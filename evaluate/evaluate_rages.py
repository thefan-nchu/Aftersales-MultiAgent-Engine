import json
import os

from datasets import Dataset
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from config import OPENAI_API_KEY, DASHSCOPE_API_KEY


def run_ragas_scoring():
    # 读取数据
    data_path = "../data/eval_dataset.json"
    if not os.path.exists(data_path):
        print("错误：找不到数据文件 data/eval_dataset.json")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # 数据格式校验与清洗
    if isinstance(eval_data, list) and len(eval_data) > 0 and isinstance(eval_data[0], list):
        eval_data = eval_data[0]
    valid_data = [item for item in eval_data if isinstance(item, dict)]
    dataset = Dataset.from_list(valid_data)

    judge_llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        base_url="https://api.chatanywhere.tech/v1",
        # 很多代理不支持 n > 1，强制 LLM 只产生一个结果
        model_kwargs={"n": 1}
    )

    # Ragas 计算相关性时需要的向量模型。
    judge_embeddings = DashScopeEmbeddings(
        model='text-embedding-v3',
        dashscope_api_key=DASHSCOPE_API_KEY
    )

    print(f"准备评估 {len(valid_data)} 条数据...")
    print("正在启动『AI 大法官』进行语义审计...")

    # 执行评估：显式传入 llm 和 embeddings
    try:
        score_result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=judge_llm,
            embeddings=judge_embeddings
        )
    except Exception as e:
        print(f"评估过程中出现异常: {e}")
        return

    import numpy as np

    print("\n" + "=" * 60)
    print("📊 手机售后系统 RAG 自动化评估报告")
    print("-" * 60)
    print(f"回答忠实度 (Faithfulness): {np.mean(score_result['faithfulness']):.4f}")
    print(f"答案相关性 (Relevancy): {np.mean(score_result['answer_relevancy']):.4f}")
    print(f"上下文精准度 (Context Precision): {np.mean(score_result['context_precision']):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_ragas_scoring()
