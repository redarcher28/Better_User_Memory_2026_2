
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
from jcards import *

def test_basic_operations():
    """测试基本操作"""
    print("=== 测试 Jcards 库模块基本操作 ===")
    
    # 创建服务实例
    service = JcardService()
    
    # 创建 SourceRef
    source_ref = SourceRef(
        conversation_id="conv_123",
        turn_range=[5, 7]
    )
    
    # 创建 Jcard
    jcard = Jcard(
        card_id="card_001",
        fact_key="passport.expiry_date",
        value={"date": "2026-10-10"},
        content="护照将于2026年10月10日过期",
        backstory="用户在对话中提供了护照过期日期",
        person="用户",
        relationship="个人信息",
        status=JcardStatus.ACTIVE,
        confidence=0.9,
        source_ref=source_ref,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # 测试写入操作
    write_ops = CardWriteOps(
        op=WriteOpType.UPSERT,
        card=jcard
    )
    
    result = service.apply_card_write_ops(write_ops)
    print(f"写入结果: {result.applied}")
    print(f"写入的卡片ID: {result.written_card_ids}")
    
    # 测试查询
    query = JcardQuery(
        person="用户",
        fact_keys=["passport.expiry_date"],
        status_in=[JcardStatus.ACTIVE],
        min_confidence=0.5
    )
    
    jcards = service.query_relevant_jcards(query)
    print(f"查询结果数量: {len(jcards)}")
    
    # 测试获取最新
    latest = service.get_latest_by_fact_key("用户", "passport.expiry_date")
    print(f"最新卡片: {latest.fact_key if latest else '无'}")
    
    # 测试按引用读取
    refs = [JcardRef(card_id="card_001")]
    ref_jcards = service.read_jcards_by_refs(refs)
    print(f"按引用读取数量: {len(ref_jcards)}")
    
    # 测试删除
    delete_req = DeleteRequest(card_ids=["card_001"])
    delete_result = service.logical_delete_cards(delete_req)
    print(f"删除数量: {delete_result.deleted_count}")
    
    print("=== 测试完成 ===")

def test_version_management():
    """测试版本管理"""
    print("\n=== 测试版本管理 ===")
    
    service = JcardService()
    
    # 创建第一版卡片
    source_ref1 = SourceRef(conversation_id="conv_1", turn_range=[1, 1])
    jcard1 = Jcard(
        card_id="card_v1",
        fact_key="user.age",
        value={"age": 25},
        content="用户年龄25岁",
        backstory="用户自述",
        person="用户",
        relationship="个人信息",
        status=JcardStatus.ACTIVE,
        confidence=0.8,
        source_ref=source_ref1,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # 创建第二版卡片（更新）
    source_ref2 = SourceRef(conversation_id="conv_1", turn_range=[3, 3])
    jcard2 = Jcard(
        card_id="card_v2",
        fact_key="user.age",
        value={"age": 26},
        content="用户年龄26岁",
        backstory="用户更正年龄",
        person="用户",
        relationship="个人信息",
        status=JcardStatus.ACTIVE,
        confidence=0.9,
        source_ref=source_ref2,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # 第一次写入
    write_ops1 = CardWriteOps(op=WriteOpType.UPSERT, card=jcard1)
    result1 = service.apply_card_write_ops(write_ops1)
    print(f"第一版写入: {result1.applied}")
    
    # 第二次写入（supersede第一版）
    write_ops2 = CardWriteOps(
        op=WriteOpType.SUPERSEDE,
        card=jcard2,
        target_card_id="card_v1"
    )
    result2 = service.apply_card_write_ops(write_ops2)
    print(f"第二版写入并supersede第一版: {result2.applied}")
    print(f"被supersede的卡片: {result2.superseded_card_ids}")
    
    # 查询最新版本
    latest = service.get_latest_by_fact_key("用户", "user.age")
    if latest:
        print(f"最新年龄: {latest.value['age']}")
    
    print("=== 版本管理测试完成 ===")

if __name__ == "__main__":
    test_basic_operations()
    test_version_management()
