import uuid
import threading
import copy
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from ..core.models import *


@dataclass
class TransactionState:
    """事务状态，用于回滚"""
    original_cards: Dict[str, Optional[Jcard]]  # card_id -> 原始卡片（None表示新增）
    original_indices: Dict[str, Any]  # 索引快照
    
    def __init__(self):
        self.original_cards = {}
        self.original_indices = {}


class JcardRepository:
    """Jcard 存储仓库，支持事务和乐观锁"""
    
    def __init__(self):
        # 主存储：card_id -> Jcard
        self._cards: Dict[str, Jcard] = {}
        
        # 索引：person_fact_key -> [card_id]（按更新时间排序）
        self._person_fact_index: Dict[Tuple[str, str], List[str]] = {}
        
        # 状态索引：status -> [card_id]
        self._status_index: Dict[JcardStatus, List[str]] = {
            JcardStatus.ACTIVE: [],
            JcardStatus.SUPERSEDED: [],
            JcardStatus.UNCERTAIN: [],
            JcardStatus.DELETED: []
        }
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 当前事务状态栈
        self._transaction_stack: List[TransactionState] = []
    
    # ========== 事务支持 ==========
    
    @contextmanager
    def transaction(self):
        """事务上下文管理器"""
        self.begin_transaction()
        try:
            yield
            self.commit_transaction()
        except Exception as e:
            self.rollback_transaction()
            raise e
    
    def begin_transaction(self):
        """开始事务"""
        with self._lock:
            state = TransactionState()
            # 不需要立即做快照，在修改时记录原始状态
            self._transaction_stack.append(state)
    
    def commit_transaction(self):
        """提交事务"""
        with self._lock:
            if not self._transaction_stack:
                raise RuntimeError("No transaction to commit")
            self._transaction_stack.pop()
    
    def rollback_transaction(self):
        """回滚事务"""
        with self._lock:
            if not self._transaction_stack:
                raise RuntimeError("No transaction to rollback")
            
            state = self._transaction_stack.pop()
            
            # 恢复卡片
            for card_id, original_card in state.original_cards.items():
                if original_card is None:
                    # 删除新增的卡片
                    if card_id in self._cards:
                        card = self._cards[card_id]
                        self._remove_from_indices(card)
                        del self._cards[card_id]
                else:
                    # 恢复原始卡片
                    self._cards[card_id] = original_card
                    self._rebuild_indices_for_card(card_id)
            
            # 恢复索引（如果有独立索引快照）
            if state.original_indices:
                self._restore_indices(state.original_indices)
    
    def _record_card_state(self, card_id: str):
        """记录卡片修改前的状态"""
        if self._transaction_stack:
            state = self._transaction_stack[-1]
            if card_id not in state.original_cards:
                original = self._cards.get(card_id)
                # 使用深拷贝复制原始卡片
                state.original_cards[card_id] = copy.deepcopy(original) if original else None
    
    def _save_with_version(self, jcard: Jcard, expected_version: Optional[int] = None) -> str:
        """
        保存卡片，支持乐观锁
        
        Args:
            jcard: 要保存的卡片
            expected_version: 期望的版本号，None表示不检查版本
            
        Returns:
            卡片ID
            
        Raises:
            ConcurrentModificationError: 版本冲突
        """
        with self._lock:
            # 检查版本冲突
            existing = self._cards.get(jcard.card_id)
            if existing and expected_version is not None:
                if existing.version != expected_version:
                    raise ConcurrentModificationError(
                        f"Card {jcard.card_id} version conflict: "
                        f"expected {expected_version}, got {existing.version}"
                    )
            
            # 记录事务状态
            if self._transaction_stack:
                self._record_card_state(jcard.card_id)
            
            # 递增版本号
            if existing:
                jcard.version = existing.version + 1
            else:
                jcard.version = 0
            
            # 更新索引和存储
            self._update_indexes(jcard)
            self._cards[jcard.card_id] = jcard
            
            return jcard.card_id
    
    def save(self, jcard: Jcard, expected_version: Optional[int] = None) -> str:
        """保存或更新 Jcard，支持乐观锁"""
        try:
            return self._save_with_version(jcard, expected_version)
        except ConcurrentModificationError:
            # 重新抛出，上层应处理
            raise
    
    def _update_indexes(self, jcard: Jcard):
        """更新所有索引"""
        # 从旧索引中移除（如果是更新）
        if jcard.card_id in self._cards:
            old_jcard = self._cards[jcard.card_id]
            self._remove_from_indices(old_jcard)
        
        # 添加到新索引
        self._add_to_indices(jcard)
    
    def _remove_from_indices(self, jcard: Jcard):
        """从所有索引中移除卡片"""
        # 从状态索引中移除
        if jcard.status in self._status_index and jcard.card_id in self._status_index[jcard.status]:
            self._status_index[jcard.status].remove(jcard.card_id)
        
        # 从 person_fact 索引中移除
        key = (jcard.person, jcard.fact_key)
        if key in self._person_fact_index and jcard.card_id in self._person_fact_index[key]:
            self._person_fact_index[key].remove(jcard.card_id)
    
    def _add_to_indices(self, jcard: Jcard):
        """添加到所有索引"""
        # 状态索引
        if jcard.status not in self._status_index:
            self._status_index[jcard.status] = []
        if jcard.card_id not in self._status_index[jcard.status]:
            self._status_index[jcard.status].append(jcard.card_id)
        
        # person_fact 索引
        key = (jcard.person, jcard.fact_key)
        if key not in self._person_fact_index:
            self._person_fact_index[key] = []
        
        if jcard.card_id not in self._person_fact_index[key]:
            self._person_fact_index[key].append(jcard.card_id)
    
    def _rebuild_indices_for_card(self, card_id: str):
        """重建单个卡片的所有索引"""
        if card_id in self._cards:
            card = self._cards[card_id]
            # 先从所有可能的位置移除
            for status in JcardStatus:
                if card_id in self._status_index[status]:
                    self._status_index[status].remove(card_id)
            
            # 从所有person_fact索引中移除
            for key in list(self._person_fact_index.keys()):
                if card_id in self._person_fact_index[key]:
                    self._person_fact_index[key].remove(card_id)
            
            # 重新添加
            self._add_to_indices(card)
    
    def _restore_indices(self, indices_snapshot: Dict[str, Any]):
        """恢复索引快照"""
        # 这里可以扩展以支持完整的索引恢复
        pass
    
    def get_with_version(self, card_id: str) -> Tuple[Optional[Jcard], int]:
        """获取卡片及其版本号"""
        with self._lock:
            card = self._cards.get(card_id)
            version = card.version if card else -1
            return card, version
    
    # ========== 查询方法（线程安全） ==========
    
    def find_by_id(self, card_id: str) -> Optional[Jcard]:
        """根据 ID 查找 Jcard"""
        with self._lock:
            return self._cards.get(card_id)
    
    def find_by_person_and_fact_key(self, person: str, fact_key: str) -> List[Jcard]:
        """根据 person 和 fact_key 查找 Jcard"""
        with self._lock:
            key = (person, fact_key)
            if key not in self._person_fact_index:
                return []
            
            card_ids = self._person_fact_index[key]
            cards = [self._cards[card_id] for card_id in card_ids if card_id in self._cards]
            
            # 按更新时间降序排序
            cards.sort(key=lambda x: x.updated_at, reverse=True)
            return cards
    
    def find_active_by_person_and_fact_key(self, person: str, fact_key: str) -> Optional[Jcard]:
        """查找指定 person 和 fact_key 的 active Jcard"""
        cards = self.find_by_person_and_fact_key(person, fact_key)
        active_cards = [c for c in cards if c.status == JcardStatus.ACTIVE and not c.deleted]
        return active_cards[0] if active_cards else None
    
    def query(self, query: JcardQuery) -> List[Jcard]:
        """根据查询条件查找 Jcard"""
        with self._lock:
            results = []
            
            for card in self._cards.values():
                if card.deleted:
                    continue
                
                # 过滤条件
                if card.person != query.person:
                    continue
                
                if query.fact_keys and card.fact_key not in query.fact_keys:
                    continue
                
                if card.status not in query.status_in:
                    continue
                
                if card.confidence < query.min_confidence:
                    continue
                
                if query.time_window:
                    window_start = datetime.fromisoformat(query.time_window.start)
                    window_end = datetime.fromisoformat(query.time_window.end)
                    if not (window_start <= card.updated_at <= window_end):
                        continue
                
                results.append(card)
                
                # 限制数量
                if len(results) >= query.limit:
                    break
            
            # 按更新时间降序排序
            results.sort(key=lambda x: x.updated_at, reverse=True)
            return results
    
    def find_by_refs(self, refs: List[JcardRef]) -> List[Jcard]:
        """根据引用查找 Jcard"""
        with self._lock:
            cards = []
            
            for ref in refs:
                if ref.card_id in self._cards:
                    card = self._cards[ref.card_id]
                    # 如果指定了 fact_key，需要匹配
                    if ref.fact_key is None or card.fact_key == ref.fact_key:
                        cards.append(card)
            
            return cards
    
    def mark_as_superseded(self, old_card_id: str, new_card_id: str, expected_version: Optional[int] = None) -> bool:
        """
        标记旧卡为 superseded，支持乐观锁
        
        Args:
            old_card_id: 旧卡片ID
            new_card_id: 新卡片ID（用于建立链接）
            expected_version: 期望的旧卡片版本号，None表示不检查
            
        Returns:
            是否成功
            
        Raises:
            ConcurrentModificationError: 版本冲突
        """
        with self._lock:
            if old_card_id not in self._cards or new_card_id not in self._cards:
                return False
            
            old_card = self._cards[old_card_id]
            
            # 检查版本冲突
            if expected_version is not None and old_card.version != expected_version:
                raise ConcurrentModificationError(
                    f"Card {old_card_id} version conflict during supersede: "
                    f"expected {expected_version}, got {old_card.version}"
                )
            
            # 记录事务状态
            if self._transaction_stack:
                self._record_card_state(old_card_id)
            
            # 更新旧卡状态
            old_card.status = JcardStatus.SUPERSEDED
            old_card.superseded_by = new_card_id
            old_card.updated_at = datetime.now()
            old_card.version += 1
            
            # 更新索引
            self._update_indexes(old_card)
            
            return True
    
    def logical_delete(self, card_ids: List[str]) -> int:
        """逻辑删除 Jcard（设置状态为DELETED）"""
        with self._lock:
            deleted_count = 0
            
            for card_id in card_ids:
                if card_id in self._cards:
                    # 记录事务状态
                    if self._transaction_stack:
                        self._record_card_state(card_id)
                    
                    card = self._cards[card_id]
                    card.deleted = True
                    card.status = JcardStatus.DELETED
                    card.updated_at = datetime.now()
                    card.version += 1
                    self._update_indexes(card)
                    deleted_count += 1
            
            return deleted_count
    
    def deactivate(self, card_id: str, expected_version: Optional[int] = None) -> bool:
        """
        停用卡片（逻辑删除），支持乐观锁
        
        Args:
            card_id: 卡片ID
            expected_version: 期望的卡片版本号，None表示不检查
            
        Returns:
            是否成功
            
        Raises:
            ConcurrentModificationError: 版本冲突
        """
        with self._lock:
            if card_id not in self._cards:
                return False
            
            card = self._cards[card_id]
            
            # 检查版本冲突
            if expected_version is not None and card.version != expected_version:
                raise ConcurrentModificationError(
                    f"Card {card_id} version conflict during deactivate: "
                    f"expected {expected_version}, got {card.version}"
                )
            
            # 记录事务状态
            if self._transaction_stack:
                self._record_card_state(card_id)
            
            # 更新卡片状态
            card.deleted = True
            card.status = JcardStatus.DELETED
            card.updated_at = datetime.now()
            card.version += 1
            
            # 更新索引
            self._update_indexes(card)
            
            return True
    
    def logical_delete_by_source(
        self, conversation_id: str, turn_id: Optional[int] = None, turn_range: Optional[List[int]] = None
    ) -> int:
        """根据来源逻辑删除 Jcard"""
        with self._lock:
            deleted_count = 0
            
            for card in self._cards.values():
                if card.source_ref.conversation_id != conversation_id:
                    continue
                if turn_id is not None and card.source_ref.turn_id != turn_id:
                    continue
                if turn_range:
                    if len(turn_range) >= 2:
                        start, end = int(turn_range[0]), int(turn_range[1])
                        if not (start <= card.source_ref.turn_id <= end):
                            continue
                    else:
                        continue
                    # 记录事务状态
                    if self._transaction_stack:
                        self._record_card_state(card.card_id)
                    
                    card.deleted = True
                    card.status = JcardStatus.DELETED
                    card.updated_at = datetime.now()
                    card.version += 1
                    self._update_indexes(card)
                    deleted_count += 1
            
            return deleted_count
    
    def get_all_active(self) -> List[Jcard]:
        """获取所有 active 的 Jcard"""
        with self._lock:
            return [self._cards[card_id] for card_id in self._status_index[JcardStatus.ACTIVE] 
                    if card_id in self._cards and not self._cards[card_id].deleted]
    
    def get_stats(self) -> Dict[str, int]:
        """获取存储统计信息"""
        with self._lock:
            total = len(self._cards)
            active = len([c for c in self._cards.values() if c.status == JcardStatus.ACTIVE and not c.deleted])
            superseded = len([c for c in self._cards.values() if c.status == JcardStatus.SUPERSEDED])
            uncertain = len([c for c in self._cards.values() if c.status == JcardStatus.UNCERTAIN])
            deleted = len([c for c in self._cards.values() if c.deleted])
            
            return {
                "total": total,
                "active": active,
                "superseded": superseded,
                "uncertain": uncertain,
                "deleted": deleted
            }
    
    def find_by_fact_key(self, fact_key: str) -> List[Jcard]:
        """
        根据fact_key查找所有Jcard（不区分person）
        对应曹合智的get_active_and_uncertain_cards_by_fact_key
        """
        with self._lock:
            results = []
            for card in self._cards.values():
                if card.deleted:
                    continue
                if card.fact_key == fact_key:
                    results.append(card)
            
            # 按更新时间降序排序
            results.sort(key=lambda x: x.updated_at, reverse=True)
            return results
    
    def find_by_entity_key(self, entity_key: str) -> List[Jcard]:
        """
        根据实体键查询Jcard
        实体键可以是：用户ID、对话ID等
        对应写入模块的query_cards_by_entity_key
        """
        with self._lock:
            results = []
            
            for card in self._cards.values():
                if card.deleted:
                    continue
                
                # 根据entity_key前缀判断查询类型
                if entity_key.startswith("user_") or entity_key.startswith("person_"):
                    # 用户查询
                    if card.person == entity_key or card.person == entity_key.replace("user_", "").replace("person_", ""):
                        results.append(card)
                elif entity_key.startswith("conv_"):
                    # 对话查询
                    if card.source_ref.conversation_id == entity_key:
                        results.append(card)
                elif entity_key.startswith("card_"):
                    # 卡片ID查询
                    if card.card_id == entity_key:
                        results.append(card)
                else:
                    # 默认按person查询
                    if card.person == entity_key:
                        results.append(card)
            
            # 按更新时间降序排序
            results.sort(key=lambda x: x.updated_at, reverse=True)
            return results
    
    def clear(self):
        """清空所有存储（主要用于测试）"""
        with self._lock:
            self._cards.clear()
            self._person_fact_index.clear()
            for status in JcardStatus:
                self._status_index[status].clear()
            self._transaction_stack.clear()