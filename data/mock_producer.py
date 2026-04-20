# data/mock_producer.py
import asyncio
import json
import os

import redis.asyncio as redis

from config.load_key import load_key

load_key()
REDIS_URL = os.environ.get("REDIS_URL")


async def produce_tickets():
    # 连接 Redis
    r = redis.from_url(REDIS_URL, decode_responses=True)

    # 加载 50 条测试数据
    with open("../data/test_cases_db.json", "r", encoding="utf-8") as f:
        tickets = json.load(f)[:3]

    print(f"开始分发 {len(tickets)} 条售后工单...")

    for ticket in tickets:
        # 将工单序列化并推入 Redis 列表（队列）
        await r.lpush("aftersales_tickets_queue", json.dumps(ticket, ensure_ascii=False))
        print(f"已排队工单: {ticket['test_id']} | 订单: {ticket['order_id']}")
        # 模拟真实间隔，每秒推入 2 条
        await asyncio.sleep(2)

    print("所有测试用例已进入待处理队列。")


if __name__ == "__main__":
    asyncio.run(produce_tickets())
