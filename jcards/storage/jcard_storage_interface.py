"""
JcardStorage 接口定义

"""
from abc import ABC, abstractmethod
from typing import Optional
from .jcard_repository import JcardRepository
from ..core.models import CardWriteOps, WriteResult, Jcard


class JcardStorage(ABC):
    """
    Jcards存储层抽象接口
    业务逻辑层与底层存储层的解耦层
    
    核心方法：
    1. apply_card_write_ops - 执行写入操作
    2. get_jcard_by_id - 维护查询
    3. get_repository - 访问底层仓库
    """
    
    @abstractmethod
    def apply_card_write_ops(self, ops: CardWriteOps) -> WriteResult:
        """
        执行卡片写入操作（新增/替代/关联）
        将决策好的CardWriteOps落地为实际卡片存储
        
        参数:
            ops: 写入操作集合
            
        返回:
            写入结果（是否成功、写入/替代的卡片ID、错误信息）
        """
        pass
    
    @abstractmethod
    def get_jcard_by_id(self, card_id: str) -> Optional[Jcard]:
        """
        根据卡片ID查询Jcard卡片
        
        参数:
            card_id: 卡片ID
            
        返回:
            Jcard对象，如果不存在则返回None
        """
        pass
    
    @abstractmethod
    def get_repository(self) -> JcardRepository:
        """获取底层存储repository（用于扩展操作）"""
        pass


# 工厂函数
_jcard_storage_instance = None

def get_jcard_storage(repository: Optional[JcardRepository] = None) -> JcardStorage:
    """获取JcardStorage实例（单例模式）"""
    global _jcard_storage_instance
    if _jcard_storage_instance is None:
        from .jcard_storage_impl import JcardStorageImpl
        _jcard_storage_instance = JcardStorageImpl(repository)
    elif repository is not None:
        # 如果传入了新的repository，重新创建实例
        from .jcard_storage_impl import JcardStorageImpl
        _jcard_storage_instance = JcardStorageImpl(repository)
    return _jcard_storage_instance