import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from ..core.models import *


class JcardRepository:
    """Jcard 存储仓库"""
    
    def __init__(self):
        # 主存储：card_id -> Jcard
        self._cards: Dict[str, Jcard] = {}
        
        # 索引：person_fact_key -> [card_id]（按更新时间排序）
        self._person_fact_index: Dict[Tuple[str, str], List[str]] = {}
        
        # 状态索引：status -> [card_id]
        self._status_index: Dict[JcardStatus, List[str]] = {
            JcardStatus.ACTIVE: [],
            JcardStatus.SUPERSEDED: [],
            JcardStatus.UNCERTAIN: []
        }
    
    def save(self, jcard: Jcard) -> str:
        """保存或更新 Jcard"""
        # 更新索引
        self._update_indexes(jcard)
        
        # 存储卡片
        self._cards[jcard.card_id] = jcard
        
        return jcard.card_id
    
    def _update_indexes(self, jcard: Jcard):
        """更新所有索引"""
        # 从旧索引中移除（如果是更新）
        if jcard.card_id in self._cards:
            old_jcard = self._cards[jcard.card_id]
            
            # 从状态索引中移除
            if old_jcard.card_id in self._status_index[old_jcard.status]:
                self._status_index[old_jcard.status].remove(old_jcard.card_id)
            
            # 从 person_fact 索引中移除
            key = (old_jcard.person, old_jcard.fact_key)
            if key in self._person_fact_index and old_jcard.card_id in self._person_fact_index[key]:
                self._person_fact_index[key].remove(old_jcard.card_id)
        
        # 添加到新索引
        # 状态索引
        if jcard.card_id not in self._status_index[jcard.status]:
            self._status_index[jcard.status].append(jcard.card_id)
        
        # person_fact 索引
        key = (jcard.person, jcard.fact_key)
        if key not in self._person_fact_index:
            self._person_fact_index[key] = []
        
        if jcard.card_id not in self._person_fact_index[key]:
            self._person_fact_index[key].append(jcard.card_id)
    
    def find_by_id(self, card_id: str) -> Optional[Jcard]:
        """根据 ID 查找 Jcard"""
        return self._cards.get(card_id)
    
    def find_by_person_and_fact_key(self, person: str, fact_key: str) -> List[Jcard]:
        """根据 person 和 fact_key 查找 Jcard"""
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
        cards = []
        
        for ref in refs:
            if ref.card_id in self._cards:
                card = self._cards[ref.card_id]
                # 如果指定了 fact_key，需要匹配
                if ref.fact_key is None or card.fact_key == ref.fact_key:
                    cards.append(card)
        
        return cards
    
    def mark_as_superseded(self, old_card_id: str, new_card_id: str) -> bool:
        """标记旧卡为 superseded"""
        if old_card_id not in self._cards or new_card_id not in self._cards:
            return False
        
        old_card = self._cards[old_card_id]
        old_card.status = JcardStatus.SUPERSEDED
        old_card.superseded_by = new_card_id
        old_card.updated_at = datetime.now()
        
        # 更新索引
        self._update_indexes(old_card)
        
        return True
    
    def logical_delete(self, card_ids: List[str]) -> int:
        """逻辑删除 Jcard"""
        deleted_count = 0
        
        for card_id in card_ids:
            if card_id in self._cards:
                card = self._cards[card_id]
                card.deleted = True
                card.updated_at = datetime.now()
                deleted_count += 1
        
        return deleted_count
    
    def logical_delete_by_source(self, conversation_id: str, turn_range: List[int]) -> int:
        """根据来源逻辑删除 Jcard"""
        deleted_count = 0
        
        for card in self._cards.values():
            if (card.source_ref.conversation_id == conversation_id and 
                card.source_ref.turn_range == turn_range):
                card.deleted = True
                card.updated_at = datetime.now()
                deleted_count += 1
        
        return deleted_count
    
    def get_all_active(self) -> List[Jcard]:
        """获取所有 active 的 Jcard"""
        return [self._cards[card_id] for card_id in self._status_index[JcardStatus.ACTIVE] 
                if card_id in self._cards and not self._cards[card_id].deleted]
    
    def get_stats(self) -> Dict[str, int]:
        """获取存储统计信息"""
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