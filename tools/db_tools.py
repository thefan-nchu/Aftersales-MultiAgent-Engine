# 查询订单、物流的函数
# 模拟数据库：实际项目中这里可以是 SQL 或 API 调用

# tools/db_tools.py
import json
import os

# 定义数据库文件路径
DB_PATH = r'G:\Agent\基于多智能体的电商售后智能路由与决策系统\data\orders_db.json'


async def _load_json_db():
    """内部工具：加载 JSON 模拟数据库"""
    if not os.path.exists(DB_PATH):
        print(f"警告：找不到数据库文件 {DB_PATH}，请确保 generate_orders.py 已运行。")
        return {}
    with open(DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


async def get_logistics_info(order_id: str):
    """从订单库中提取物流状态"""
    db = await _load_json_db()
    order = db.get(str(order_id))
    if not order:
        return None

    # 为了保持兼容性，我们将订单中的 status 包装成 logistics 字典格式
    return {
        "status": order.get("status"),
        "delivery_date": order.get("purchase_date") if order.get("status") == "已签收" else None,
        "current_location": "配送中心" if order.get("status") == "运输中" else "客户所在地"
    }


async def get_order_financial_details(order_id: str):
    """获取财务明细"""
    db = await _load_json_db()
    return db.get(str(order_id))


async def get_combined_order_info(order_id: str):
    """
    组合查询：一次性返回全量数据供 Agent 决策
    """
    db = await _load_json_db()
    order = db.get(str(order_id))

    if not order:
        return None

    # 计算或提取签收日期
    # 在你的 mock 结构中，如果已签收，我们将 purchase_date 视为三包起算点
    return {
        "order_id": order_id,
        "product_name": order["product_name"],
        "purchase_date": order["purchase_date"],
        "delivery_date": order["purchase_date"] if order["status"] == "已签收" else None,
        "pay_amount": order["pay_amount"],
        "status": order["status"],
        "logistics_status": order["status"]
    }
