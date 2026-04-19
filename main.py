# main.py
import asyncio
import json

import redis.asyncio as redis
from langchain_core.messages import HumanMessage, AIMessage

from config import REDIS_URL
from core.graph import create_after_sales_graph

# 设置最大并发数。如果你的 API 额度高，可以设为 10-20；如果显存压力大或 API 受限，设为 1-5。
MAX_CONCURRENT_TASKS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)


async def handle_ticket(app, ticket_data):
    """
    单个工单的异步处理协程
    """
    # 使用信号量控制并发，防止瞬间挤爆 API 或内存
    async with semaphore:
        test_id = ticket_data.get("test_id", "Unknown")
        order_id = ticket_data.get("order_id")
        user_input = ticket_data.get("user_input", "")
        image_path = ticket_data.get("image_path")

        print(f"[任务启动] ID: {test_id} | 正在分析订单: {order_id}")

        # --- 构造多模态消息内容 ---
        content = [{"type": "text", "text": user_input}]
        if image_path:
            # 这里传的是本地路径，vision_node 内部会根据模式选择读取图片还是读取 Mock
            content.append({"type": "image_url", "image_url": {"url": image_path}})

        # --- 构造初始 State ---
        # 必须包含 test_metadata 字段，vision_node 才能进行 Mock 拦截
        initial_state = {
            "messages": [HumanMessage(content=content)],
            "order_id": order_id,
            "test_metadata": ticket_data  # 注入全量测试数据，包含 mock_vision_output
        }

        # 构造 LangGraph 配置 (thread_id 用于 Redis 持久化隔离)
        config = {"configurable": {"thread_id": f"session_{test_id}"}}

        try:
            # 运行异步图
            # ainvoke 会自动遍历所有节点直至 END
            result = await app.ainvoke(initial_state, config=config)

            # 提取执行结果
            final_messages = result.get("messages", [])
            decision = result.get("final_decision", "N/A")

            # 获取最后一条 AI 的回复内容
            ai_reply = "系统未生成有效回复"
            ai_messages = [m for m in final_messages if isinstance(m, AIMessage)]
            for msg in reversed(ai_messages):
                # 必须是 AIMessage 且 content 不能为空
                if isinstance(msg, AIMessage) and msg.content:
                    # 检查 content 是否是列表（多模态 AI 偶尔会返回列表），统一转为字符串
                    if isinstance(msg.content, list):
                        ai_reply = " ".join([p.get("text", "") for p in msg.content if p.get("type") == "text"])
                    else:
                        ai_reply = msg.content

                    # 检查是否是 supervisor 产生的 JSON（我们不想把 JSON 打印给终端用户看）
                    if ai_reply.startswith("{") and "next_agent" in ai_reply:
                        continue  # 跳过 JSON，继续往前找专家的回复
                    break  # 找到了专家的“人话”回复，退出循环

            print(f"[处理成功] ID: {test_id} | 最终决策: {decision}")
            print(f"系统回复摘要: {ai_reply.replace('\n', ' ').strip()}")

        except Exception as e:
            print(f"[处理异常] ID: {test_id} | 错误信息: {str(e)}")


async def main_engine():
    """
    主引擎：持续监听 Redis 队列并派发任务
    """
    # 初始化持久化 Graph（含 Redis Checkpointer）
    app = await create_after_sales_graph()

    # 初始化 Redis 连接
    await clear_test_sessions(redis_url=REDIS_URL)
    r = redis.from_url(REDIS_URL, decode_responses=True)

    print("\n" + "=" * 60)
    print("手机售后智能调度引擎 (v2.0 异步版) 已就绪")
    print(f"正在监听队列: 'aftersales_tickets_queue' | 最大并发: {MAX_CONCURRENT_TASKS}")
    print("=" * 60 + "\n")

    while True:
        try:
            # 使用阻塞式弹出 (BRPOP)，超时时间 5 秒
            # 这保证了 CPU 不会空转，没有任务时会自动挂起
            raw_data = await r.brpop("aftersales_tickets_queue", timeout=5)

            if raw_data:
                # raw_data 格式为 (key_name, value)
                ticket_payload = json.loads(raw_data[1])

                # 非阻塞派发
                # 使用 create_task 立即启动协程，然后主循环立刻回到 brpop 等待下一个工单
                asyncio.create_task(handle_ticket(app, ticket_payload))

        except Exception as loop_e:
            print(f" 引擎循环遇到错误: {loop_e}")
            await asyncio.sleep(2)  # 遇到连接错误时稍作停顿


async def clear_test_sessions(redis_url):
    r = redis.from_url(redis_url)
    keys = await r.keys("checkpoint*")
    if keys:
        await r.delete(*keys)


if __name__ == "__main__":
    try:
        asyncio.run(main_engine())
    except KeyboardInterrupt:
        print("\n售后引擎已安全停止。")
