"""
向量库服务模块（构建 + 维护 + 检索接口）

默认实现：SQLite（本地磁盘持久化）+ Numpy（余弦相似度 TopK）

说明（为什么不用 Chroma）：
- Windows + Python 3.13 常见缺少 chroma-hnswlib 预编译轮子，pip 会要求本机 MSVC 编译工具。
- 为了“开箱即用”，这里选用零编译依赖的 SQLite 方案；后续要切换到 Chroma/Milvus 只需替换这一层。

对外能力：
  - upsert_records：写入/更新 chunk（embedding 由上游提供）
  - similarity_search：RAG 检索连接点（query_embedding + filters -> hits）
  - logical_delete：按 chunk_id 或 conversation_id 逻辑删除（deleted=true）
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    conversation_id: str
    turn_start: int
    turn_end: int
    timestamp_start: str | None = None
    timestamp_end: str | None = None
    participants: list[str] = Field(default_factory=list)
    intent_tag: str | None = None
    source: str | None = None
    chunk_version: int = 1
    deleted: bool = False


class ChunkRecord(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


def deterministic_chunk_id(
    conversation_id: str, turn_start: int, turn_end: int, chunk_version: int
) -> str:
    raw = f"{conversation_id}|{turn_start}|{turn_end}|v{chunk_version}".encode("utf-8")
    return "chk_" + hashlib.sha1(raw).hexdigest()  # short enough, deterministic


def _default_persist_dir() -> str:
    root = Path(__file__).resolve().parents[1]
    return str(root / ".vector_store")


DEFAULT_VECTOR_STORE_DIR = _default_persist_dir()


@dataclass
class VectorStoreConfig:
    persist_dir: str = field(default_factory=_default_persist_dir)
    db_file: str = "vector_store.sqlite3"


class SQLiteVectorStoreService:
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        persist = Path(self.config.persist_dir)
        persist.mkdir(parents=True, exist_ok=True)
        self._db_path = persist / self.config.db_file
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              chunk_id TEXT PRIMARY KEY,
              text TEXT NOT NULL,
              embedding BLOB NOT NULL,
              embedding_dim INTEGER NOT NULL,
              metadata_json TEXT NOT NULL,
              conversation_id TEXT NOT NULL,
              turn_start INTEGER NOT NULL,
              turn_end INTEGER NOT NULL,
              deleted INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_conv_deleted ON chunks(conversation_id, deleted);"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_deleted ON chunks(deleted);")
        self._conn.commit()

    def upsert_records(
        self,
        records: List[ChunkRecord],
        embeddings: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        if embeddings is None:
            raise ValueError("SQLiteVectorStoreService requires embeddings to upsert_records")
        if len(embeddings) != len(records):
            raise ValueError("embeddings length must match records length")

        cur = self._conn.cursor()
        for r, e in zip(records, embeddings):
            vec = np.asarray(e, dtype=np.float32)
            meta = r.metadata.model_dump()
            deleted = 1 if meta.get("deleted") else 0
            cur.execute(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, text, embedding, embedding_dim, metadata_json, conversation_id, turn_start, turn_end, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.chunk_id,
                    r.text,
                    vec.tobytes(),
                    int(vec.shape[0]),
                    json.dumps(meta, ensure_ascii=False),
                    meta["conversation_id"],
                    int(meta["turn_start"]),
                    int(meta["turn_end"]),
                    deleted,
                ),
            )
        self._conn.commit()
        return {"upserted": len(records), "errors": []}

    def similarity_search(
        self,
        query_text: str,
        top_k: int = 6,
        filters: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        where = filters or {}
        deleted = bool(where.get("deleted", False))
        conversation_id = where.get("conversation_id")

        if query_embedding is None:
            raise ValueError("similarity_search requires query_embedding in this implementation")

        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)

        sql = "SELECT chunk_id, text, embedding, embedding_dim, metadata_json FROM chunks WHERE deleted = ?"
        params: List[Any] = [1 if deleted else 0]
        if conversation_id:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)

        rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return {"query_text": query_text, "top_k": top_k, "filters": where, "hits": []}

        # Validate embedding dimension against stored vectors (assumes single embedding model per store)
        expected_dim = int(rows[0][3])  # embedding_dim
        if int(q.shape[0]) != expected_dim:
            raise ValueError(
                f"query_embedding dim mismatch: got {int(q.shape[0])}, expected {expected_dim}. "
                "Use the same embedding model as ingestion."
            )

        ids: List[str] = []
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        vecs: List[np.ndarray] = []
        for chunk_id, text, blob, dim, meta_json in rows:
            v = np.frombuffer(blob, dtype=np.float32, count=int(dim))
            v = v / (np.linalg.norm(v) + 1e-12)
            ids.append(chunk_id)
            texts.append(text)
            metas.append(json.loads(meta_json))
            vecs.append(v)

        M = np.stack(vecs, axis=0)
        scores = M @ q
        top_idx = np.argsort(-scores)[:top_k]

        hits: List[SearchHit] = []
        for i in top_idx:
            ii = int(i)
            hits.append(
                SearchHit(
                    chunk_id=ids[ii],
                    score=float(scores[ii]),
                    text=texts[ii],
                    metadata=metas[ii],
                )
            )

        return {
            "query_text": query_text,
            "top_k": top_k,
            "filters": where,
            "hits": [h.model_dump() for h in hits],
        }

    def fetch_records_by_chunk_ids(self, chunk_ids: List[str]) -> List[ChunkRecord]:
        if not chunk_ids:
            return []
        placeholders = ",".join(["?"] * len(chunk_ids))
        sql = (
            "SELECT chunk_id, text, metadata_json FROM chunks "
            f"WHERE chunk_id IN ({placeholders})"
        )
        rows = self._conn.execute(sql, chunk_ids).fetchall()
        records: List[ChunkRecord] = []
        for chunk_id, text, meta_json in rows:
            meta = json.loads(meta_json)
            records.append(ChunkRecord(chunk_id=chunk_id, text=text, metadata=ChunkMetadata(**meta)))
        return records

    def logical_delete_by_chunk_ids(self, chunk_ids: List[str]) -> Dict[str, Any]:
        if not chunk_ids:
            return {"deleted": 0, "errors": []}
        cur = self._conn.cursor()
        cur.executemany(
            "UPDATE chunks SET deleted = 1 WHERE chunk_id = ?",
            [(cid,) for cid in chunk_ids],
        )
        self._conn.commit()
        return {"deleted": len(chunk_ids), "errors": []}

    def logical_delete_by_conversation_id(self, conversation_id: str) -> Dict[str, Any]:
        cur = self._conn.cursor()
        cur.execute("UPDATE chunks SET deleted = 1 WHERE conversation_id = ?", (conversation_id,))
        self._conn.commit()
        return {"deleted": cur.rowcount if cur.rowcount != -1 else 0, "errors": []}


def _demo() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    svc = SQLiteVectorStoreService(VectorStoreConfig(persist_dir=".vector_store_demo"))
    record = ChunkRecord(
        chunk_id=deterministic_chunk_id("conv1", 1, 1, 1),
        text="[上下文：demo] 我护照 2026-10-10 过期。",
        metadata=ChunkMetadata(
            conversation_id="conv1",
            turn_start=1,
            turn_end=1,
            participants=["用户"],
            intent_tag="profile_update",
            chunk_version=1,
            deleted=False,
        ),
    )
    svc.upsert_records([record], embeddings=[[0.0, 1.0, 0.0]])  # demo only
    print(json.dumps({"ok": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _demo()

