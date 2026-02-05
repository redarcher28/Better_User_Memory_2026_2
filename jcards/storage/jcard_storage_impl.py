"""
JcardStorage 接口实现

"""
from typing import Optional
from .jcard_storage_interface import JcardStorage
from .jcard_repository import JcardRepository
from ..core.models import CardWriteOps, WriteResult, Jcard
from ..api.jcard_service import JcardService


class JcardStorageImpl(JcardStorage):
    """JcardStorage接口的具体实现"""
    
    def __init__(self, repository: Optional[JcardRepository] = None):
        self.repository = repository or JcardRepository()
        self.service = JcardService(self.repository)
    
    def apply_card_write_ops(self, ops: CardWriteOps) -> WriteResult:
        """
        执行卡片写入操作
        调用JcardService的apply_card_write_ops方法
        """
        return self.service.apply_card_write_ops(ops)
    
    def get_jcard_by_id(self, card_id: str) -> Optional[Jcard]:
        """
        根据卡片ID查询Jcard卡片
        """
        return self.repository.find_by_id(card_id)
    
    def get_repository(self) -> JcardRepository:
        """获取底层存储repository"""
        return self.repository