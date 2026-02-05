
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import json
import uuid


class ConcurrentModificationError(Exception):
    """乐观锁版本冲突异常"""
    pass


class JcardStatus(str, Enum):
    """Jcard状态枚举"""
    ACTIVE = "active"
    SUPERSEDED = "superseded"  # 逻辑删除+被替代
    UNCERTAIN = "uncertain"
    DELETED = "deleted"  # 显式逻辑删除


class WriteOpType(str, Enum):
    """写入操作类型枚举"""
    UPSERT = "upsert"      # 插入或更新
    SUPERSEDE = "supersede"  # 替代旧卡
    CORRECT = "correct"    # 纠正：逻辑删除旧卡+写新卡
    DEACTIVATE = "deactivate"  # 逻辑删除


@dataclass
class SourceRef:
    """来源引用，用于审计追溯"""
    conversation_id: str  # 会话ID
    turn_id: int  # 单个回合ID（用于精确追溯）
    speaker: str  # 说话者标识
    timestamp: datetime  # 事件时间戳
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceRef':
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


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
    # 稳定标识：event_id基于来源信息生成，确保可追溯
    card_id: str  # 业务标识
    fact_key: str
    value: Dict[str, Any]
    content: str
    backstory: str
    person: str
    relationship: str
    status: JcardStatus
    confidence: float
    source_ref: SourceRef  # 必须包含conversation_id, turn_id, speaker, timestamp
    created_at: datetime
    updated_at: datetime
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 稳定事件标识
    version: int = 0  # 版本号，用于乐观锁
    superseded_by: Optional[str] = None
    deleted: bool = False
    
    def __post_init__(self):
        """后初始化：确保event_id基于来源信息生成，确保稳定性"""
        # 如果event_id是默认生成的uuid，则基于source_ref生成稳定标识
        if self.event_id and len(self.event_id) == 36:  # UUID格式
            # 检查是否是UUID格式，如果是，则重新生成稳定的event_id
            src = self.source_ref
            self.event_id = f"{src.conversation_id}_{src.turn_id}_{src.speaker}_{int(src.timestamp.timestamp())}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        # 递归处理source_ref
        data['source_ref'] = self.source_ref.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Jcard':
        """从字典创建 Jcard 对象"""
        data = data.copy()
        data['status'] = JcardStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        # 处理source_ref
        if isinstance(data['source_ref'], dict):
            data['source_ref'] = SourceRef.from_dict(data['source_ref'])
        else:
            # 已经是SourceRef对象
            pass
        # 处理version字段（向后兼容）
        if 'version' not in data:
            data['version'] = 0
        # 处理event_id（向后兼容）
        if 'event_id' not in data:
            # 尝试基于source_ref生成
            src = data['source_ref']
            data['event_id'] = f"{src.conversation_id}_{src.turn_id}_{src.speaker}_{int(src.timestamp.timestamp())}"
        return cls(**data)
    
    def generate_stable_card_id(self) -> str:
        """生成稳定的卡片ID，基于事件ID和事实键"""
        # 使用event_id和fact_key生成稳定的card_id
        # 这样可以确保相同来源的相同事实具有相同的card_id
        return f"{self.event_id}_{self.fact_key}"


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
        # 处理 source_ref 的时间戳序列化
        if isinstance(self.source_ref, SourceRef):
            data['source_ref'] = self.source_ref.to_dict()
        else:
            sr = data.get('source_ref')
            if isinstance(sr, dict) and isinstance(sr.get('timestamp'), datetime):
                sr['timestamp'] = sr['timestamp'].isoformat()
                data['source_ref'] = sr
        return data


@dataclass
class CardWriteOps:
    """卡片写入操作"""
    op: WriteOpType
    card: Jcard
    target_card_id: Optional[str] = None  # 用于supersede/link操作
    # 版本检查字段（向后兼容）
    expected_version: Optional[int] = None  # 乐观锁期望版本（已废弃，建议使用下面的字段）
    card_expected_version: Optional[int] = None  # 新卡片的期望版本
    target_expected_version: Optional[int] = None  # 目标卡片的期望版本
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['op'] = self.op.value
        data['card'] = self.card.to_dict()
        # 不序列化None值，以保持字典简洁
        if self.expected_version is not None:
            data['expected_version'] = self.expected_version
        if self.card_expected_version is not None:
            data['card_expected_version'] = self.card_expected_version
        if self.target_expected_version is not None:
            data['target_expected_version'] = self.target_expected_version
        return data


@dataclass
class WriteResult:
    """写入操作结果"""
    applied: bool
    upserted_ids: List[str] = field(default_factory=list)  # 新增或更新的卡片ID
    updated_ids: List[str] = field(default_factory=list)   # 被更新的卡片ID（仅更新，不包括新增）
    superseded_ids: List[str] = field(default_factory=list)  # 被替代的卡片ID（同superseded_card_ids，但更名为复数形式）
    deleted_ids: List[str] = field(default_factory=list)   # 被逻辑删除的卡片ID
    errors: List[str] = field(default_factory=list)
    
    # 为了向后兼容，我们保留以下属性，但使用property
    @property
    def written_card_ids(self) -> List[str]:
        """返回所有被写入的卡片ID（包括新增和更新）"""
        return self.upserted_ids + self.updated_ids
    
    @property
    def superseded_card_ids(self) -> List[str]:
        """返回被替代的卡片ID（兼容旧版本）"""
        return self.superseded_ids
    
    @written_card_ids.setter
    def written_card_ids(self, value: List[str]):
        """设置写入的卡片ID，将同时设置upserted_ids和updated_ids为空，并添加value到upserted_ids"""
        self.upserted_ids = value
        self.updated_ids = []
    
    @superseded_card_ids.setter
    def superseded_card_ids(self, value: List[str]):
        """设置被替代的卡片ID"""
        self.superseded_ids = value


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
    turn_id: Optional[int] = None
    turn_range: Optional[List[int]] = None


@dataclass
class DeleteResult:
    """删除结果"""
    deleted_count: int
    failed_ids: List[str]
    errors: List[str]
