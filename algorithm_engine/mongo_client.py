"""
algorithm_engine.mongo_client
──────────────────────────────
MongoDB 客户端封装（可选）。如果 pymongo 未安装或连接失败，返回 None，
调用方检查后降级存储到 MySQL JSON 字段。
"""

import logging
from functools import lru_cache

logger = logging.getLogger('algorithm_engine')

_client = None


def get_mongo_db():
    """获取 MongoDB database 对象。失败时返回 None。"""
    global _client
    try:
        if _client is None:
            import pymongo
            from django.conf import settings
            _client = pymongo.MongoClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=3000,
            )
            # 触发连接测试
            _client.admin.command('ping')
        from django.conf import settings
        return _client[settings.MONGODB_DATABASE]
    except Exception as e:
        logger.warning('MongoDB 连接失败（将跳过详情存储）: %s', e)
        return None


def save_detection_detail(collection_name: str, document: dict) -> str | None:
    """
    将检测详情保存到 MongoDB。
    返回插入的文档 ID（str），如果 MongoDB 不可用则返回 None。
    """
    db = get_mongo_db()
    if db is None:
        return None
    try:
        result = db[collection_name].insert_one(document)
        return str(result.inserted_id)
    except Exception as e:
        logger.warning('MongoDB 写入失败: %s', e)
        return None


def get_detection_detail(collection_name: str, doc_id: str) -> dict | None:
    """按 doc_id 查询 MongoDB 详情文档。"""
    db = get_mongo_db()
    if db is None:
        return None
    try:
        from bson import ObjectId
        return db[collection_name].find_one({'_id': ObjectId(doc_id)})
    except Exception as e:
        logger.warning('MongoDB 查询失败: %s', e)
        return None
