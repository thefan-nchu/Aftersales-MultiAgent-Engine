# 检索向量库的函数

# tools/rag_tools.py
from datetime import datetime

from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore

from config import REDIS_URL

embedding_model = DashScopeEmbeddings(model='text-embedding-v3')
# 保存向量数据库

config = RedisConfig(
    index_name="phone_agent_policies",
    redis_url=REDIS_URL,
)

vector_store = RedisVectorStore(embedding_model, config=config)

CURRENT_TIME = datetime(2024, 5, 20)

reranker = DashScopeRerank(
    model='gte-rerank-v2',
    top_n=1,
)


def _parse_warranty_days(condition_str: str) -> int:
    """内部工具：解析政策中的时间限制字符串"""
    if "7天" in condition_str: return 7
    if "15天" in condition_str: return 15
    if "180天" in condition_str: return 180
    if "1年" in condition_str or "在保" in condition_str: return 365
    if "2年" in condition_str: return 730
    return 9999  # 默认长期有效


async def get_policy_decision(user_text: str, vision_tags: list, purchase_date_str: str,
                              category: str = None):
    """
    RAG 核心工具：
    1. 语义检索最相关的政策案例
    2. 根据购机日期进行二次逻辑校验
    """
    # 构造组合查询语句（提升特征匹配度）
    try:
        # purchase_date_str 可能来自 delivery_date "2024-05-17 14:00:00"
        # 截取后变成 "2024-05-17"，完美匹配 %Y-%m-%d
        purchase_date = datetime.strptime(purchase_date_str[:10], "%Y-%m-%d")
        user_days = (CURRENT_TIME - purchase_date).days
    except Exception as e:
        print(f"日期解析失败: {purchase_date_str}, 错误: {e}")
        # 如果解析失败，使用当前时间（即认为刚买，对用户最宽容）
        purchase_date = datetime.now()
        user_days = CURRENT_TIME

    # 动态构造检索词：如果有标签用标签，没标签全靠文字
    if vision_tags and len(vision_tags) > 0:
        query_parts = [user_text]
        if vision_tags:
            query_parts.append(f"特征: {' '.join(vision_tags)}")
        search_query = " ".join(query_parts)
    else:
        search_query = user_text

    # 使用 similarity_search_with_score 可以获取匹配得分，用于阈值过滤
    results = await vector_store.asimilarity_search_with_score(
        search_query,
        k=10,
    )

    # 重排序
    docs_to_rerank = [res[0] for res in results]
    reranked_docs = reranker.compress_documents(
        documents=docs_to_rerank,
        query=search_query
    )

    # 获取最匹配的一个案例
    best_doc = reranked_docs[0]
    meta = best_doc.metadata

    # 提取政策中原始的阈值和决策
    w_days = int(meta.get("warranty_days", 9999))
    original_decision = meta.get("decision")
    final_decision = original_decision  # 默认结论

    # 执行决策降级逻辑
    # 如果用户的购机天数超过了政策规定的保修天数
    if user_days > w_days:
        # 定义降级规则矩阵
        if original_decision in ["REFUND_FULL", "REPLACE_NEW"]:
            # 如果原本是退换货，但超过了时效（例如>15天），降级为保内维修
            # 如果连保修也过了（>365天），则降级为付费维修
            final_decision = "FREE_REPAIR" if user_days <= 365 else "PAID_REPAIR"

        elif original_decision == "FREE_REPAIR":
            # 如果原本是免费维修，但超过了1年，降级为自费维修
            final_decision = "PAID_REPAIR"

        elif original_decision == "PAID_REPAIR":
            # 原本就是自费的，保持不变
            final_decision = "PAID_REPAIR"

    return {
        "policy_id": meta["id"],
        "final_decision": final_decision,
        "policy_content": best_doc.page_content,
        "days_since_purchase": user_days,
    }
