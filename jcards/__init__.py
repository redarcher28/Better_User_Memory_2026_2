
__version__ = "2"
__author__ = "Anqi"

# 1. 核心数据模型
from .core.models import *

# 2. JcardStorage接口
from .storage.jcard_storage_interface import JcardStorage, get_jcard_storage

# 3. 服务层（内部使用）
from .api.jcard_service import JcardService, get_jcard_service

__all__ = [
    
    'JcardStorage',
    'get_jcard_storage',
    
    # 数据模型
    'Jcard',
    'JcardView',
    'JcardStatus',
    'WriteOpType',
    'SourceRef',
    'TimeWindow',
    'CardWriteOps',
    'WriteResult',
    'JcardQuery',
    'JcardRef',
    'GetJcardsRequest',
    'DeleteRequest',
    'DeleteResult',
    
    # 内部服务
    'JcardService',
    'get_jcard_service'
]