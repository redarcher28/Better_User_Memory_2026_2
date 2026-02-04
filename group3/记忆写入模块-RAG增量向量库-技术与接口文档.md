### 文档目的
本文件定义「向量库服务模块（构建 + 维护 + 检索接口）」的职责、数据模型、接口契约与关键工程约束（增量、可追溯、可删除/可更新、可复现 chunk）。

该模块对下游仅提供 **RAG 记忆检索功能块连接向量库的查询接口** 与必要的写入/删除维护接口；不负责任何“回答生成/最终输出”。

---

### 模块定位（What）
**向量库服务模块**负责两件事：
- **构建与维护**：将对话事件流加工为可检索的 **ChunkRecord**（对话片段 + 元数据 + embedding），并以增量方式 upsert 到向量库/索引中（含索引与元数据过滤能力）。
- **提供检索连接点**：暴露给下游 RAG/Agent 的**相似性检索接口**（输入查询/过滤条件，返回匹配到的 chunk 记录），作为“证据检索底座”。

---

### 职责边界（Responsibilities / Non-goals）
- **负责**
  - 对话文本清洗（轻量去噪/归一化，保证可检索）。
  - 增量切分 chunk（支持重叠、语义优先）。
  - 生成**上下文前缀摘要**（时间/人物/意图），并固化到 chunk 文本或元数据。
  - 生成 embedding 并 upsert 入向量库（含索引与元数据）。
  - 提供 **RAG 连接向量库的检索查询接口**（相似性检索 + 元数据过滤）。
  - 提供按 `conversation_id/turn_range` 的逻辑删除或更新能力（最少支持 logical delete）。
- **不负责**
  - 不负责 rerank、prompt 组装或“最终输出”（这些由下游 RAG 记忆检索功能块 / Agent Core 完成）。
  - 不负责 Jcards 生成（由另一模块完成）。
  - 不负责复杂权限裁决（可提供 metadata，具体 enforcement 由上层做）。

---

### 核心概念与数据模型

#### 1) ChunkRecord（向量库条目）
一个可检索"证据单元"，包含稳定 ID、文本与元数据。**支持双向对话记录**（用户输入 + LLM回复）。

```json
{
  "chunk_id": "chk_9b21...",
  "text": "[上下文：2026-02-02 用户在讨论护照信息] 用户: 我护照是 2025-02-18 过期。助手: 了解，您的护照将于2025年2月18日过期。请问您是否有出行计划？",
  "metadata": {
    "conversation_id": "conv_abc",
    "turn_range": [18, 19],
    "timestamp_range": ["2026-02-02T10:15:30+08:00", "2026-02-02T10:15:32+08:00"],
    "participants": ["用户"],
    "speakers": ["user", "assistant"],
    "intent_tag": "profile_update",
    "chunk_version": 1,
    "deleted": false
  }
}
```

字段说明（关键）：
- **chunk_id**：必须稳定且可复现（便于 upsert/更新/删除）。
- **metadata.turn_range + conversation_id**：必须存在（可追溯证据来源）。
- **metadata.speakers**：记录该chunk中包含的角色（["user", "assistant"]），便于区分用户输入与LLM回复。
- **chunk_version**：切分策略变化时递增，避免新旧混淆。
- **deleted**：逻辑删除标记（检索时过滤）。

#### 2) chunk_id 生成建议（可复现）
```text
chunk_id = hash(conversation_id + start_turn + end_turn + chunk_version)
```

---

### 增量策略（推荐实现路径）
#### 1) Append-only + 逻辑删除（优先推荐）
- 新事件到来：只新增 chunk（upsert）。
- 用户要求删除/撤回：将命中的 chunk 标记 `deleted=true`（检索过滤）。

优点：实现简单、稳定、利于审计；缺点：需要后台定期物理清理（可选）。

#### 2) 更新（可选增强）
当需要替换某段文本（例如摘要策略升级），使用相同 `chunk_id` upsert 覆盖 `text/metadata`，或提升 `chunk_version` 重建。

---

### 接口定义（Service APIs）
说明：这里的“输出”仅指**接口返回值**（给调用方程序使用），不是面向终端用户的最终输出。

#### 接口 1：build_incremental_chunks（构建增量 Chunk）
将一组新增对话事件构造成 chunk 列表。
- **输入**：`MemoryEvent[]`（同一 conversation 的连续 turn 段）
- **输出**：`ChunkRecord[]`

```json
{
  "conversation_id": "conv_abc",
  "turn_range": [15, 20],
  "chunks": [ { "chunk_id": "...", "text": "...", "metadata": {} } ]
}
```

#### 接口 2：upsert_vector_records（向量库写入/更新）
- **输入**：`ChunkRecord[]`（服务内部完成 embedding 与入库；调用方不需要处理向量）
- **输出**：写入结果（成功/失败、upsert 数量）

```json
{
  "upserted": 12,
  "updated": 0,
  "errors": []
}
```

#### 接口 3：logical_delete_by_source（逻辑删除）
按来源范围删除（满足“撤回/隐私删除”需求）。
- **输入**：`conversation_id` + `turn_range`（或 chunk_id 列表）
- **输出**：删除结果

```json
{
  "deleted": 3,
  "errors": []
}
```

#### 接口 4：embed_db（RAG 连接向量库的检索接口）
面向下游 RAG 记忆检索功能块的核心查询接口。
- **输入**：`query_text` + `top_k` +（可选）`filters`
  - filters 常见字段：`conversation_id`、`participants`、`timestamp_range`、`deleted=false`（默认必须过滤）
- **输出**：匹配结果列表（ChunkRecord 视图：`chunk_id/text/metadata`），供下游做 rerank、拼 prompt、引用证据。

```json
{
  "query_text": "东京之行要准备什么",
  "top_k": 6,
  "filters": { "deleted": false },
  "hits": [
    {
      "chunk_id": "chk_...",
      "score": 0.82,
      "text": "[上下文：...] ...",
      "metadata": { "conversation_id": "conv_abc", "turn_range": [12, 15], "chunk_version": 1 }
    }
  ]
}
```

#### 接口 5：embed_chunk（查询向量化接口）
面向记忆查询层的向量化接口。将用户输入的查询字符串转换为向量表示。
- **输入**：`query` (string) - 用户提示词/查询内容
- **输出**：`List[float]` - 归一化的向量表示

```python
# Python 调用示例
from rag_ingest_incremental import embed_chunk

# 方式1：使用便捷函数
vector = embed_chunk("我护照什么时候过期？")
print(f"向量维度: {len(vector)}")  # 512 (BAAI/bge-small-zh-v1.5)

# 方式2：使用服务类（推荐，支持批量）
from rag_ingest_incremental import EmbeddingService

service = EmbeddingService.get_instance()
vector = service.embed_chunk("东京之行要准备什么？")

# 批量向量化（性能优化）
queries = ["护照过期日期", "签证要求", "行程安排"]
vectors = service.embed_batch(queries)
```

**特性**：
- 单例模式：模型只加载一次，避免重复开销
- 自动归一化：返回的向量已归一化，适合余弦相似度计算
- 线程安全：支持多线程并发调用
- 默认模型：BAAI/bge-small-zh-v1.5（512维中文向量）

---

### 上下文前缀摘要（Context Prefix）规范
目的：解决对话分块后语义不完整问题，使 chunk 在孤立检索时仍可理解。

最小字段建议：
- 时间（或时间范围）
- 参与人
- 当时在讨论的主题/意图（短句）

示例：
```text
[上下文：2026-02-02 用户在讨论东京行程与证件准备]
```

---

### 幂等与一致性要求
- **同一 (conversation_id, turn_range, chunk_version)** 重跑必须生成相同 chunk_id 集合。
- upsert 必须可重放（重复调用不应产生重复记录）。
- 检索侧必须过滤 `deleted=true`。

---

### 性能与实现建议（不绑定具体技术栈）
- **异步化**：Write API 同步落原文事件；Index Worker 异步切分/摘要/embedding/upsert。
- **批处理**：embedding 与 upsert 按 batch 做（降低成本与延迟）。
- **监控指标（建议最小）**：入库 chunk 数、embedding 延迟、upsert 失败率、按会话的 backlog。

---

### 与上游/下游的契约
- **上游（事件流/对话系统/记忆写入编排）**：提供有序的 `MemoryEvent`，至少包含 `conversation_id/turn_id/text/timestamp/speaker`。
  - **重要**：`speaker` 字段支持 `"user"`（用户输入）和 `"assistant"`（LLM回复），**两者都需要写入向量库**以支持完整对话上下文检索。
  - 调用 `build_incremental_chunks` 与 `upsert_vector_records` 完成增量构建。
- **下游（RAG 记忆检索功能块）**：调用 `embed_db` 获取 `hits`（chunk + 来源元数据），作为后续 rerank/证据引用的输入。
- **底座（向量库实现）**：支持按 `chunk_id` upsert；支持按 metadata 过滤（至少 `conversation_id` + `deleted` + `speakers`）；支持返回 `chunk_id + text + metadata` 作为 Evidence。

