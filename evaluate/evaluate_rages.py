import json
import os

from datasets import Dataset
from langchain_community.embeddings import DashScopeEmbeddings  # 复用你之前的 embeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from config import OPENAI_API_KEY, DASHSCOPE_API_KEY


def run_ragas_scoring():
    # 读取数据
    data_path = "../data/eval_dataset.json"
    if not os.path.exists(data_path):
        print("❌ 错误：找不到数据文件 data/eval_dataset.json")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # 数据格式校验与清洗
    if isinstance(eval_data, list) and len(eval_data) > 0 and isinstance(eval_data[0], list):
        eval_data = eval_data[0]
    valid_data = [item for item in eval_data if isinstance(item, dict)]
    dataset = Dataset.from_list(valid_data)

    # 2. 【核心修复】：显式配置裁判 LLM
    # 重点：配置代理地址，并处理 "requested 3 generations" 的问题
    judge_llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        base_url="https://api.chatanywhere.tech/v1",
        # 很多代理不支持 n > 1，强制 LLM 只产生一个结果
        model_kwargs={"n": 1}
    )

    # 3. 【核心修复】：显式配置评估用的 Embeddings
    # Ragas 计算相关性时需要向量模型。如果不配置，它会默认调 OpenAI 官方接口导致 401
    judge_embeddings = DashScopeEmbeddings(
        model='text-embedding-v3',
        dashscope_api_key=DASHSCOPE_API_KEY  # 确保环境变量里有这个
    )

    print(f"🚀 准备评估 {len(valid_data)} 条数据...")
    print("⚖️ 正在启动『AI 大法官』进行语义审计 (预计耗时 1-3 分钟)...")

    # 4. 执行评估：显式传入 llm 和 embeddings
    try:
        score_result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=judge_llm,
            embeddings=judge_embeddings
        )
    except Exception as e:
        print(f"❌ 评估过程中出现异常: {e}")
        # 如果还是报 Key 错误，打印一下环境变量看看
        # print(f"Current Key: {os.environ.get('OPENAI_API_KEY')}")
        return

    # 5. 分析并生成报表
    df = score_result.to_pandas()

    # 业务决策准确率比对
    if 'system_decision' in df.columns and 'expected_decision' in df.columns:
        df['is_decision_ok'] = df.apply(
            lambda x: 1 if str(x['system_decision']) == str(x['expected_decision']) else 0,
            axis=1
        )
        decision_acc = df['is_decision_ok'].mean()
    else:
        decision_acc = 0.0

    print("\n" + "=" * 60)
    print("📊 手机售后系统 RAG 自动化评估报告")
    print("-" * 60)
    print(f"判定准确率 (Decision Accuracy): {decision_acc * 100:.2f}%")
    print(f"回答忠实度 (Faithfulness): {score_result['faithfulness']:.4f}")
    print(f"答案相关性 (Relevancy): {score_result['answer_relevancy']:.4f}")
    print(f"上下文精准度 (Context Precision): {score_result['context_precision']:.4f}")
    print("=" * 60)

    # 保存结果
    output_path = "../data/evaluation_final_report.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ 详细评估结果已保存至: {output_path}")


if __name__ == "__main__":
    run_ragas_scoring()
