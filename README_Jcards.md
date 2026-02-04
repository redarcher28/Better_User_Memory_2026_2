
# Jcards 库模块

结构化事实存储系统，为用户记忆提供可信的核心事实底座。

## 功能特性

- **结构化存储**: 存储结构化的用户事实（Jcards）
- **版本管理**: 支持事实的版本化和状态管理（active/superseded/uncertain）
- **审计追溯**: 完整的来源追踪和变更记录
- **多维度查询**: 支持按事实键、人员、时间窗口、置信度等多维度查询
- **幂等写入**: 支持幂等操作，避免重复处理
- **隐私保护**: 支持软删除，保护用户隐私

## 接口规范

### 1. 写入接口
- `apply_card_write_ops(ops: CardWriteOps, idempotency_key: Optional[str]) -> WriteResult`

### 2. 查询接口
- `query_relevant_jcards(query: JcardQuery) -> List[JcardView]`
- `get_latest_by_fact_key(person: str, fact_key: str) -> Optional[JcardView]`
- `read_jcards_by_refs(refs: List[JcardRef]) -> List[JcardView]`

### 3. 删除接口
- `logical_delete_cards(request: DeleteRequest) -> DeleteResult`

### 4. 工具接口
- `get_Jcards_to_string(request: GetJcardsRequest) -> str`

## 快速开始

```python
from jcards import get_jcard_service, JcardQuery, JcardStatus

# 获取服务实例
service = get_jcard_service()

# 查询相关卡片
query = JcardQuery(
    person="用户",
    fact_keys=["passport.expiry_date"],
    status_in=[JcardStatus.ACTIVE],
    min_confidence=0.5
)

cards = service.query_relevant_jcards(query)