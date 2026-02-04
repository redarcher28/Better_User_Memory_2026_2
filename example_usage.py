
from datetime import datetime
from jcards import *

def example_for_detector():
    """为 Detector 模块提供的使用示例"""
    print("=== Detector 模块使用示例 ===")
    
    # 获取服务实例
    service = get_jcard_service()
    
    # 假设 Detector 需要查询相关事实
    query = JcardQuery(
        person="用户",
        fact_keys=["passport.expiry_date", "trip.departure_date"],
        status_in=[JcardStatus.ACTIVE, JcardStatus.UNCERTAIN],
        min_confidence=0.6,
        limit=10
    )
    
    # 查询相关卡片
    relevant_cards = service.query_relevant_jcards(query)
    print(f"Detector 查询到 {len(relevant_cards)} 张相关卡片")
    
    for card in relevant_cards[:3]:  # 显示前3张
        print(f"  - {card.fact_key}: {card.value} (置信度: {card.confidence})")

def example_for_upstream():
    """为上游生成模块提供的使用示例"""
    print("\n=== 上游生成模块使用示例 ===")
    
    service = get_jcard_service()
    
    # 模拟从对话中提取的事实
    source_ref = SourceRef(
        conversation_id="conv_20240204_001",
        turn_range=[10, 12]
    )
    
    new_jcard = Jcard(
        card_id=f"card_{datetime.now().timestamp()}",
        fact_key="flight.booking_reference",
        value={"ref": "ABC123", "airline": "Air China"},
        content="用户预订了国航航班，预订号ABC123",
        backstory="用户在对话中提供了航班预订信息",
        person="用户",
        relationship="旅行信息",
        status=JcardStatus.ACTIVE,
        confidence=0.85,
        source_ref=source_ref,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # 创建写入操作
    write_ops = CardWriteOps(
        op=WriteOpType.UPSERT,
        card=new_jcard
    )
    
    # 执行写入（带幂等键）
    result = service.apply_card_write_ops(
        write_ops, 
        idempotency_key=f"conv_20240204_001_turn_10_12"
    )
    
    if result.applied:
        print(f"成功写入卡片: {new_jcard.fact_key}")
    else:
        print(f"写入失败: {result.errors}")

def example_for_evidence_module():
    """为证据生成模块提供的使用示例"""
    print("\n=== 证据生成模块使用示例 ===")
    
    service = get_jcard_service()
    
    # 需要作为证据的卡片引用
    evidence_refs = [
        JcardRef(card_id="card_001"),
        JcardRef(fact_key="passport.expiry_date", card_id="card_001")
    ]
    
    # 批量读取卡片作为证据
    evidence_cards = service.read_jcards_by_refs(evidence_refs)
    
    print(f"获取到 {len(evidence_cards)} 张证据卡片")
    for card in evidence_cards:
        print(f"证据: {card.fact_key} = {card.value}")

def main():
    """主函数"""
    print("Jcards 库模块使用示例")
    print("=" * 50)
    
    example_for_detector()
    example_for_upstream()
    example_for_evidence_module()
    
    print("\n" + "=" * 50)
    print("示例执行完成")

if __name__ == "__main__":
    main()
