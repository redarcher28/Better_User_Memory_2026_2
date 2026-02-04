
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..core.models import *
from ..storage.jcard_repository import JcardRepository


class JcardService:
    """
    Jcards 库模块主服务
    实现所有技术文档中定义的接口
    """
    
    def __init__(self, repository: Optional[JcardRepository] = None):
        self.repository = repository or JcardRepository()
        self._idempotency_keys = set()  # 简单的幂等键管理
    
    # -------------------- 接口1：apply_card_write_ops --------------------
    def apply_card_write_ops(self, ops: CardWriteOps, idempotency_key: Optional[str] = None) -> WriteResult:
        """
        应用卡片写入操作
        支持幂等性（通过 idempotency_key）
        """
        # 幂等性检查
        if idempotency_key and idempotency_key in self._idempotency_keys:
            return WriteResult(
                applied=True,
                written_card_ids=[],
                superseded_card_ids=[],
                errors=["操作已执行（幂等）"]
            )
        
        written_card_ids = []
        superseded_card_ids = []
        errors = []
        
        try:
            if ops.op == WriteOpType.UPSERT:
                # 保存新卡
                self.repository.save(ops.card)
                written_card_ids.append(ops.card.card_id)
                
                # 如果有目标卡ID，标记为 superseded
                if ops.target_card_id:
                    if self.repository.mark_as_superseded(ops.target_card_id, ops.card.card_id):
                        superseded_card_ids.append(ops.target_card_id)
                    else:
                        errors.append(f"无法标记卡片为superseded: {ops.target_card_id}")
            
            elif ops.op == WriteOpType.SUPERSEDE:
                if ops.target_card_id and ops.card:
                    # 保存新卡
                    self.repository.save(ops.card)
                    written_card_ids.append(ops.card.card_id)
                    
                    # 标记旧卡为 superseded
                    if self.repository.mark_as_superseded(ops.target_card_id, ops.card.card_id):
                        superseded_card_ids.append(ops.target_card_id)
                    else:
                        errors.append(f"无法标记卡片为superseded: {ops.target_card_id}")
                else:
                    errors.append("supersede操作需要target_card_id和新卡片")
            
            elif ops.op == WriteOpType.LINK:
                # 链接操作：这里简单实现为保存卡片并记录关联
                self.repository.save(ops.card)
                written_card_ids.append(ops.card.card_id)
                # 实际实现中可能需要维护关联关系表
            
            else:
                errors.append(f"未知的操作类型: {ops.op}")
            
            # 记录幂等键
            if idempotency_key:
                self._idempotency_keys.add(idempotency_key)
                
        except Exception as e:
            errors.append(f"操作执行失败: {str(e)}")
        
        return WriteResult(
            applied=len(errors) == 0,
            written_card_ids=written_card_ids,
            superseded_card_ids=superseded_card_ids,
            errors=errors
        )
    
    # -------------------- 接口2：query_relevant_jcards --------------------
    def query_relevant_jcards(self, query: JcardQuery) -> List[JcardView]:
        """
        查询相关 Jcards
        返回 JcardView 列表（最小必要字段）
        """
        jcards = self.repository.query(query)
        return [JcardView.from_jcard(jcard) for jcard in jcards]
    
    # -------------------- 接口3：get_latest_by_fact_key --------------------
    def get_latest_by_fact_key(self, person: str, fact_key: str) -> Optional[JcardView]:
        """
        获取指定 person 和 fact_key 的最新 Jcard
        优先返回 active，若无则返回 uncertain
        """
        # 查找 active 卡片
        active_card = self.repository.find_active_by_person_and_fact_key(person, fact_key)
        if active_card:
            return JcardView.from_jcard(active_card)
        
        # 查找 uncertain 卡片
        all_cards = self.repository.find_by_person_and_fact_key(person, fact_key)
        uncertain_cards = [c for c in all_cards if c.status == JcardStatus.UNCERTAIN and not c.deleted]
        
        if uncertain_cards:
            # 按更新时间取最新的
            latest = max(uncertain_cards, key=lambda x: x.updated_at)
            return JcardView.from_jcard(latest)
        
        return None
    
    # -------------------- 接口4：read_jcards_by_refs --------------------
    def read_jcards_by_refs(self, refs: List[JcardRef], view_type: str = "JcardView") -> List[JcardView]:
        """
        按引用读取卡片
        支持返回完整 Jcard 或 JcardView
        """
        jcards = self.repository.find_by_refs(refs)
        
        if view_type == "JcardView":
            return [JcardView.from_jcard(jcard) for jcard in jcards]
        else:
            # 这里应该返回完整 Jcard，但接口定义返回 JcardView
            # 所以默认返回 JcardView
            return [JcardView.from_jcard(jcard) for jcard in jcards]
    
    # -------------------- 接口5：logical_delete_cards --------------------
    def logical_delete_cards(self, request: DeleteRequest) -> DeleteResult:
        """
        逻辑删除卡片
        支持按 card_ids 或 source_ref 删除
        """
        deleted_count = 0
        failed_ids = []
        errors = []
        
        try:
            if request.card_ids:
                deleted_count = self.repository.logical_delete(request.card_ids)
            elif request.conversation_id and request.turn_range:
                deleted_count = self.repository.logical_delete_by_source(
                    request.conversation_id, 
                    request.turn_range
                )
            else:
                errors.append("必须提供 card_ids 或 conversation_id 和 turn_range")
        
        except Exception as e:
            errors.append(f"删除操作失败: {str(e)}")
        
        return DeleteResult(
            deleted_count=deleted_count,
            failed_ids=failed_ids,
            errors=errors
        )
    
    # -------------------- 接口6：get_Jcards_to_string --------------------
    def get_Jcards_to_string(self, request: GetJcardsRequest) -> str:
        """
        获取 Jcards 的字符串表示
        主要用于调试和日志
        """
        # 构建查询条件
        status_in = [JcardStatus.ACTIVE]
        if request.include_superseded:
            status_in.append(JcardStatus.SUPERSEDED)
        if request.include_uncertain:
            status_in.append(JcardStatus.UNCERTAIN)
        
        query = JcardQuery(
            person=request.person,
            fact_keys=request.fact_keys,
            status_in=status_in,
            min_confidence=request.min_confidence,
            limit=0  # 不限制数量
        )
        
        jcards = self.repository.query(query)
        
        # 转换为字典列表
        result = []
        for jcard in jcards:
            jcard_dict = {
                "card_id": jcard.card_id,
                "fact_key": jcard.fact_key,
                "value": jcard.value,
                "status": jcard.status.value,
                "confidence": jcard.confidence,
                "updated_at": jcard.updated_at.isoformat(),
                "source_ref": jcard.source_ref.to_dict()
            }
            result.append(jcard_dict)
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    # -------------------- 辅助方法 --------------------
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        return self.repository.get_stats()
    
    def clear_idempotency_keys(self):
        """清空幂等键（用于测试）"""
        self._idempotency_keys.clear()


# 创建全局单例实例
_jcard_service_instance = None

def get_jcard_service() -> JcardService:
    """获取 JcardService 单例实例"""
    global _jcard_service_instance
    if _jcard_service_instance is None:
        _jcard_service_instance = JcardService()
    return _jcard_service_instance
