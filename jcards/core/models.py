
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import json


class JcardStatus(str, Enum):
    """Jcard状态枚举"""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    UNCERTAIN = "uncertain"


class WriteOpType(str, Enum):
    """写入操作类型枚举"""
    UPSERT = "upsert"
    SUPERSEDE = "supersede"
    LINK = "link"


@dataclass
class SourceRef:
    """来源引用，用于审计追溯"""
    conversation_id: str
    turn_range: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimeWindow:
    """时间窗口查询条件"""
    start: str  # ISO 格式日期字符串
    end: str    # ISO 格式日期字符串
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Jcard:
    """Jcard 存储实体（完整字段）"""
    card_id: str
    fact_key: str
    value: Dict[str, Any]
    content: str
    backstory: str
    person: str
    relationship: str
    status: JcardStatus
    confidence: float
    source_ref: SourceRef
    created_at: datetime
    updated_at: datetime
    superseded_by: Optional[str] = None
    deleted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Jcard':
        """从字典创建 Jcard 对象"""
        data = data.copy()
        data['status'] = JcardStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['source_ref'] = SourceRef(**data['source_ref'])
        return cls(**data)


@dataclass
class JcardView:
    """Jcard 读侧视图（最小必要字段）"""
    card_id: str
    fact_key: str
    value: Dict[str, Any]
    status: JcardStatus
    confidence: float
    updated_at: datetime
    source_ref: SourceRef
    
    @classmethod
    def from_jcard(cls, jcard: Jcard) -> 'JcardView':
        """从完整 Jcard 创建视图"""
        return cls(
            card_id=jcard.card_id,
            fact_key=jcard.fact_key,
            value=jcard.value,
            status=jcard.status,
            confidence=jcard.confidence,
            updated_at=jcard.updated_at,
            source_ref=jcard.source_ref
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        data = asdict(self)
        data['status'] = self.status.value
        data['updated_at'] = self.updated_at.isoformat()
        return data


@dataclass
class CardWriteOps:
    """卡片写入操作"""
    op: WriteOpType
    card: Jcard
    target_card_id: Optional[str] = None  # 用于supersede/link操作
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['op'] = self.op.value
        data['card'] = self.card.to_dict()
        return data


@dataclass
class WriteResult:
    """写入操作结果"""
    applied: bool
    written_card_ids: List[str]
    superseded_card_ids: List[str]
    errors: List[str]


@dataclass
class JcardQuery:
    """Jcard查询条件"""
    person: str
    fact_keys: List[str]
    status_in: List[JcardStatus]
    min_confidence: float
    time_window: Optional[TimeWindow] = None
    limit: int = 50


@dataclass
class JcardRef:
    """Jcard引用"""
    card_id: str
    fact_key: Optional[str] = None


@dataclass
class GetJcardsRequest:
    """获取Jcards字符串请求"""
    person: str
    fact_keys: List[str]
    include_superseded: bool = False
    include_uncertain: bool = False
    min_confidence: float = 0.0


@dataclass
class DeleteRequest:
    """删除请求"""
    card_ids: List[str]
    conversation_id: Optional[str] = None
    turn_range: Optional[List[int]] = None


@dataclass
class DeleteResult:
    """删除结果"""
    deleted_count: int
    failed_ids: List[str]
    errors: List[str]
