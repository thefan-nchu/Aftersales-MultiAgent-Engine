import json

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_redis import RedisConfig, RedisVectorStore

from config import REDIS_URL

# 1. 初始化环境与配置
embedding_model = DashScopeEmbeddings(model='text-embedding-v3')

config = RedisConfig(
    index_name="phone_agent_policies",  # 建议起个明确的索引名
    redis_url=REDIS_URL,
)

# 2. 初始化向量存储
vector_store = RedisVectorStore(embedding_model, config=config)


def _convert_condition_to_days(condition_str: str) -> int:
    """内部工具：解析政策中的时间限制字符串"""
    if "7天" in condition_str: return 7
    if "15天" in condition_str: return 15
    if "180天" in condition_str: return 180
    if "1年" in condition_str or "在保" in condition_str: return 365
    if "2年" in condition_str: return 730
    return 9999


def ingest_phone_policies(json_path: str):
    # 3. 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        policies = json.load(f)

    documents = []

    for item in policies:
        # --- 构造检索内容 (Page Content) ---
        # 我们把症状和视觉特征串联起来，这是模型最可能检索的部分
        content = f"故障症状: {item['symptom']}. 视觉特征标签: {', '.join(item['visual_feature'])}. 政策内容: {item['policy_content']}"
        # --- 构造元数据 (Metadata) ---
        # 所有的原始字段都存入元数据，方便检索到后直接读取执行逻辑
        metadata = {
            "id": item["id"],
            "category": item["category"],
            "decision": item["decision"],
            "original_condition": item["warranty_condition"],
            "warranty_days": _convert_condition_to_days(item["warranty_condition"]),
            "symptom": item["symptom"]
        }

        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    # 4. 批量添加到 Redis
    print(f"正在向 Redis 灌入 {len(documents)} 条售后政策...")
    vector_store.add_documents(documents)
    print("向量库构建完成！")


if __name__ == "__main__":
    # 执行灌库
    json_file_path = "../data/phone_policy_db.json"
    ingest_phone_policies(json_file_path)
