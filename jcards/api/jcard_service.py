
from typing import Optional, Dict, Any
import json
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
        支持幂等性（通过 idempotency_key）和事务
        
        根据操作类型返回详细结果：
        - upserted_ids: 新增的卡片ID（之前不存在）
        - updated_ids: 更新的卡片ID（已经存在）
        - superseded_ids: 被替代的卡片ID（状态变为SUPERSEDED）
        - deleted_ids: 被逻辑删除的卡片ID（状态变为DELETED）
        """
        # 幂等性检查
        if idempotency_key and idempotency_key in self._idempotency_keys:
            return WriteResult(
                applied=True,
                upserted_ids=[],
                updated_ids=[],
                superseded_ids=[],
                deleted_ids=[],
                errors=["操作已执行（幂等）"]
            )
        
        upserted_ids = []   # 新增的卡片
        updated_ids = []    # 更新的卡片（已存在）
        superseded_ids = [] # 被替代的卡片
        deleted_ids = []    # 被删除的卡片
        errors = []
        
        try:
            # 使用事务执行操作
            with self.repository.transaction():
                # 检查卡片是否存在，用于区分新增和更新
                existing_card = self.repository.find_by_id(ops.card.card_id)
                is_update = existing_card is not None and not existing_card.deleted
                
                if ops.op == WriteOpType.UPSERT:
                    # 确定新卡片的期望版本：优先使用card_expected_version，其次使用expected_version（向后兼容）
                    card_expected_version = ops.card_expected_version if ops.card_expected_version is not None else ops.expected_version
                    # 保存新卡，检查新卡片的版本（如果提供了期望版本）
                    self.repository.save(ops.card, card_expected_version)
                    if is_update:
                        updated_ids.append(ops.card.card_id)
                    else:
                        upserted_ids.append(ops.card.card_id)
                    
                    # 如果有目标卡ID，标记为 superseded，并检查目标卡片的版本
                    if ops.target_card_id:
                        # 确定目标卡片的期望版本：优先使用target_expected_version，其次使用expected_version（向后兼容）
                        target_expected_version = ops.target_expected_version if ops.target_expected_version is not None else ops.expected_version
                        if self.repository.mark_as_superseded(ops.target_card_id, ops.card.card_id, target_expected_version):
                            superseded_ids.append(ops.target_card_id)
                        else:
                            errors.append(f"无法标记卡片为superseded: {ops.target_card_id}")
                
                elif ops.op == WriteOpType.SUPERSEDE:
                    if ops.target_card_id and ops.card:
                        # 保存新卡，不检查新卡片的版本（因为新卡片可能不存在，或者我们不关心其版本）
                        # 注意：如果新卡片已经存在，我们可能希望检查其版本，但SUPERSEDE操作通常用于替换另一个卡片，所以新卡片应该是全新的。
                        # 因此，我们这里不检查新卡片的版本。
                        self.repository.save(ops.card, None)
                        upserted_ids.append(ops.card.card_id)
                        
                        # 标记旧卡为 superseded，检查目标卡片的版本
                        target_expected_version = ops.target_expected_version if ops.target_expected_version is not None else ops.expected_version
                        if self.repository.mark_as_superseded(ops.target_card_id, ops.card.card_id, target_expected_version):
                            superseded_ids.append(ops.target_card_id)
                        else:
                            errors.append(f"无法标记卡片为superseded: {ops.target_card_id}")
                    else:
                        errors.append("supersede操作需要target_card_id和新卡片")
                
                elif ops.op == WriteOpType.CORRECT:
                    # CORRECT操作：逻辑删除旧卡 + 写新卡
                    # 需要target_card_id指定要删除的旧卡
                    if ops.target_card_id and ops.card:
                        # 首先逻辑删除旧卡
                        target_expected_version = ops.target_expected_version if ops.target_expected_version is not None else ops.expected_version
                        if self.repository.deactivate(ops.target_card_id, target_expected_version):
                            deleted_ids.append(ops.target_card_id)
                        else:
                            errors.append(f"无法逻辑删除卡片: {ops.target_card_id}")
                            # 如果删除失败，我们仍然尝试写入新卡吗？按照原子性，应该回滚。
                            # 但先记录错误，稍后统一处理。
                        
                        # 然后写入新卡
                        card_expected_version = ops.card_expected_version if ops.card_expected_version is not None else ops.expected_version
                        self.repository.save(ops.card, card_expected_version)
                        if is_update:
                            updated_ids.append(ops.card.card_id)
                        else:
                            upserted_ids.append(ops.card.card_id)
                    else:
                        errors.append("CORRECT操作需要target_card_id和新卡片")
                
                elif ops.op == WriteOpType.DEACTIVATE:
                    # DEACTIVATE操作：逻辑删除指定的卡片
                    if ops.target_card_id:
                        target_expected_version = ops.target_expected_version if ops.target_expected_version is not None else ops.expected_version
                        if self.repository.deactivate(ops.target_card_id, target_expected_version):
                            deleted_ids.append(ops.target_card_id)
                        else:
                            errors.append(f"无法逻辑删除卡片: {ops.target_card_id}")
                    else:
                        errors.append("DEACTIVATE操作需要target_card_id")
                
                elif ops.op == WriteOpType.LINK:
                    # 链接操作：这里简单实现为保存卡片并记录关联
                    # 可以检查新卡片的版本（如果提供了card_expected_version）
                    card_expected_version = ops.card_expected_version if ops.card_expected_version is not None else ops.expected_version
                    self.repository.save(ops.card, card_expected_version)
                    if is_update:
                        updated_ids.append(ops.card.card_id)
                    else:
                        upserted_ids.append(ops.card.card_id)
                    # 实际实现中可能需要维护关联关系表
                
                else:
                    errors.append(f"未知的操作类型: {ops.op}")
                
                # 如果事务中有错误，抛出异常以触发回滚
                if errors:
                    raise Exception("; ".join(errors))
                
                # 记录幂等键（如果提供）
                if idempotency_key:
                    self._idempotency_keys.add(idempotency_key)
        
        except ConcurrentModificationError as e:
            errors.append(f"并发修改冲突: {str(e)}")
        except Exception as e:
            # 事务已经回滚，我们只需记录错误
            if not errors:  # 如果errors为空，说明是其他异常
                errors.append(f"操作执行失败: {str(e)}")
        
        if errors:
            upserted_ids = []
            updated_ids = []
            superseded_ids = []
            deleted_ids = []
        
        # 构建结果，使用新的字段名
        result = WriteResult(
            applied=len(errors) == 0,
            upserted_ids=upserted_ids,
            updated_ids=updated_ids,
            superseded_ids=superseded_ids,
            deleted_ids=deleted_ids,
            errors=errors
        )
        
        return result

    def get_Jcards_to_string(self, request: GetJcardsRequest) -> str:
        """
        返回符合请求条件的 Jcards JSON 字符串（供 RAG_query 使用）。
        """
        statuses = [JcardStatus.ACTIVE]
        if request.include_superseded:
            statuses.append(JcardStatus.SUPERSEDED)
        if request.include_uncertain:
            statuses.append(JcardStatus.UNCERTAIN)

        query = JcardQuery(
            person=request.person,
            fact_keys=request.fact_keys,
            status_in=statuses,
            min_confidence=request.min_confidence,
        )
        cards = self.repository.query(query)
        views = [JcardView.from_jcard(c).to_dict() for c in cards]
        return json.dumps(views, ensure_ascii=False)
    
    # -------------------- 接口2：logical_delete_cards --------------------
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
            elif request.conversation_id and (request.turn_id is not None or request.turn_range):
                deleted_count = self.repository.logical_delete_by_source(
                    request.conversation_id, 
                    turn_id=request.turn_id,
                    turn_range=request.turn_range
                )
            else:
                errors.append("必须提供 card_ids 或 conversation_id 与 turn_id/turn_range")
        
        except Exception as e:
            errors.append(f"删除操作失败: {str(e)}")
        
        return DeleteResult(
            deleted_count=deleted_count,
            failed_ids=failed_ids,
            errors=errors
        )
    
    # -------------------- 辅助方法 --------------------
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        return self.repository.get_stats()
    
    def clear_idempotency_keys(self):
        """清空幂等键（用于测试）"""
        self._idempotency_keys.clear()


# 创建全局单例实例
_jcard_service_instance = None

def get_jcard_service(repository: Optional[JcardRepository] = None) -> JcardService:
    """获取JcardService实例（支持传入自定义repository）"""
    global _jcard_service_instance
    if repository is not None:
        return JcardService(repository)
    elif _jcard_service_instance is None:
        _jcard_service_instance = JcardService()
    return _jcard_service_instance