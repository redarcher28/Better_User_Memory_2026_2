"""
增量入库：解析/归一化 -> 切分 Chunk -> 上下文前缀摘要 -> Embedding -> 入库（Chroma）

定位：
- 这是“写入侧”的 RAG 向量库构建脚本/模块。
- 读取对话事件（JSONL/JSON）后，生成 ChunkRecord 并 upsert 到向量库服务模块。

默认选型（可替换）：
- embedding: sentence-transformers (BAAI/bge-small-zh-v1.5)
- vector store: Chroma persistent (see rag_vector_store.py)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from dateutil import parser as dtparser
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from rag_vector_store import (
    SQLiteVectorStoreService,
    ChunkMetadata,
    ChunkRecord,
    VectorStoreConfig,
    deterministic_chunk_id,
)


class MemoryEvent(BaseModel):
    event_id: str
    conversation_id: str
    turn_id: int
    speaker: str  # "user" / "assistant" / etc.
    text: str
    timestamp: str | None = None
    participants: list[str] = Field(default_factory=list)
    locale: str | None = "zh-CN"


@dataclass
class ChunkingConfig:
    max_chars: int = 600
    overlap_chars: int = 80
    chunk_version: int = 1


def normalize_text(text: str) -> str:
    text = text.strip()
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # normalize common punctuation spacing (lightweight)
    text = text.replace(" ,", ",").replace(" .", ".")
    return text


def parse_time(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        dt = dtparser.parse(ts)
        # ISO format with offset if present
        return dt.isoformat()
    except Exception:
        return ts


def detect_intent_tag(text: str) -> str:
    """
    超轻量意图归类（写入侧用于上下文前缀摘要）：
    后续如果你们有更强的抽取器，可替换为上游传入 intent_tag。
    """
    t = text.lower()
    if any(k in t for k in ["护照", "签证", "证件", "过期", "续签"]):
        return "document_check"
    if any(k in t for k in ["东京", "出国", "机票", "航班", "酒店", "行程", "旅行", "旅游"]):
        return "travel_plan"
    if any(k in t for k in ["转账", "汇款", "付款", "账单", "还款"]):
        return "payment_instruction"
    if any(k in t for k in ["地址", "住址", "电话", "邮箱"]):
        return "profile_update"
    return "general"


def build_context_prefix(
    timestamp_start: str | None,
    participants: list[str],
    intent_tag: str | None,
) -> str:
    parts = []
    if timestamp_start:
        # keep only date for brevity if parseable
        try:
            dt = dtparser.parse(timestamp_start)
            parts.append(dt.strftime("%Y-%m-%d"))
        except Exception:
            parts.append(timestamp_start)
    if participants:
        parts.append("、".join(participants[:3]))
    if intent_tag:
        parts.append(f"主题:{intent_tag}")
    ctx = " ".join([p for p in parts if p])
    if not ctx:
        ctx = "unknown"
    return f"[上下文：{ctx}] "


def group_events_by_conversation(events: Iterable[MemoryEvent]) -> Dict[str, List[MemoryEvent]]:
    grouped: Dict[str, List[MemoryEvent]] = defaultdict(list)
    for e in events:
        grouped[e.conversation_id].append(e)
    for cid in grouped:
        grouped[cid].sort(key=lambda x: x.turn_id)
    return grouped


def chunk_conversation_events(
    events: List[MemoryEvent],
    cfg: ChunkingConfig,
) -> List[Tuple[int, int, str, str | None, str | None, list[str], str]]:
    """
    以 turn 为基本单位拼接，满足 max_chars 时切 chunk。
    返回：turn_start, turn_end, chunk_text, ts_start, ts_end, participants, intent_tag
    """
    chunks: List[Tuple[int, int, str, str | None, str | None, list[str], str]] = []

    # buffer stored as event-aligned lines so overlap won't break turn_range explainability
    buf_lines: List[Tuple[int, str]] = []  # (turn_id, line)
    buf_chars = 0
    ts_start: str | None = None
    last_ts: str | None = None
    participants: list[str] = []
    intent_tags: List[str] = []

    def current_turn_start() -> int:
        return buf_lines[0][0] if buf_lines else (events[0].turn_id if events else 0)

    def current_turn_end() -> int:
        return buf_lines[-1][0] if buf_lines else (events[0].turn_id if events else 0)

    def make_overlap_lines() -> List[Tuple[int, str]]:
        """Keep last lines whose char sum <= overlap_chars."""
        if cfg.overlap_chars <= 0 or not buf_lines:
            return []
        kept: List[Tuple[int, str]] = []
        total = 0
        for turn_id, line in reversed(buf_lines):
            if total + len(line) > cfg.overlap_chars and kept:
                break
            kept.append((turn_id, line))
            total += len(line)
            if total >= cfg.overlap_chars:
                break
        return list(reversed(kept))

    def flush() -> None:
        nonlocal buf_lines, buf_chars, ts_start, participants, intent_tags
        if not buf_lines:
            return
        turn_start = current_turn_start()
        turn_end = current_turn_end()

        raw_text = "\n".join([line for _, line in buf_lines])
        raw_text = normalize_text(raw_text)
        intent = intent_tags[-1] if intent_tags else detect_intent_tag(raw_text)
        chunks.append((turn_start, turn_end, raw_text, ts_start, last_ts, participants, intent))

        # event-aligned overlap
        overlap_lines = make_overlap_lines()
        buf_lines = overlap_lines
        buf_chars = sum(len(line) for _, line in buf_lines)
        intent_tags = []
        participants = []
        # ts_start becomes timestamp of first overlapped line's event; approximate with last_ts if unknown
        # We'll set ts_start on next incoming event if buffer empty; if overlap kept, keep previous ts_start.
        if not buf_lines:
            ts_start = None

    for e in events:
        last_ts = parse_time(e.timestamp)
        if ts_start is None:
            ts_start = last_ts
        if e.participants:
            participants = e.participants
        intent_tags.append(detect_intent_tag(e.text))
        line = f"{e.speaker}: {normalize_text(e.text)}"
        buf_lines.append((e.turn_id, line))
        buf_chars += len(line)
        if buf_chars >= cfg.max_chars:
            flush()

    flush()
    return chunks


def build_chunk_records(
    events: Iterable[MemoryEvent],
    chunk_cfg: ChunkingConfig,
) -> List[ChunkRecord]:
    grouped = group_events_by_conversation(events)
    records: List[ChunkRecord] = []

    for conv_id, conv_events in grouped.items():
        if not conv_events:
            continue
        chunks = chunk_conversation_events(conv_events, cfg=chunk_cfg)
        for turn_start, turn_end, raw_text, ts_start, ts_end, participants, intent_tag in chunks:
            prefix = build_context_prefix(ts_start, participants, intent_tag)
            doc = prefix + raw_text
            cid = deterministic_chunk_id(
                conversation_id=conv_id,
                turn_start=turn_start,
                turn_end=turn_end,
                chunk_version=chunk_cfg.chunk_version,
            )
            meta = ChunkMetadata(
                conversation_id=conv_id,
                turn_start=turn_start,
                turn_end=turn_end,
                timestamp_start=ts_start,
                timestamp_end=ts_end,
                participants=participants,
                intent_tag=intent_tag,
                chunk_version=chunk_cfg.chunk_version,
                deleted=False,
            )
            records.append(ChunkRecord(chunk_id=cid, text=doc, metadata=meta))

    return records


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> List[List[float]]:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # cosine space recommended
        show_progress_bar=len(texts) > 64,
    )
    # ensure python lists
    return np.asarray(emb, dtype=np.float32).tolist()


class EmbeddingService:
    """
    向量化服务：管理embedding模型，提供查询向量化接口
    
    用途：
    - 供记忆查询层调用，将用户查询转换为向量
    - 单例模式，避免重复加载模型
    """
    
    _instance: Optional['EmbeddingService'] = None
    _model: Optional[SentenceTransformer] = None
    _model_name: str = "BAAI/bge-small-zh-v1.5"
    
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """
        初始化向量化服务
        
        参数:
            model_name: SentenceTransformer模型名称（默认: BAAI/bge-small-zh-v1.5）
        """
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self) -> None:
        """加载embedding模型（懒加载）"""
        if self._model is None or self._model_name != self.model_name:
            print(f"Loading embedding model: {self.model_name}...", file=sys.stderr)
            self._model = SentenceTransformer(self.model_name)
            self._model_name = self.model_name
            print(f"Model loaded successfully.", file=sys.stderr)
    
    @classmethod
    def get_instance(cls, model_name: str = "BAAI/bge-small-zh-v1.5") -> 'EmbeddingService':
        """
        获取单例实例（避免重复加载模型）
        
        参数:
            model_name: 模型名称
        
        返回:
            EmbeddingService实例
        """
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance
    
    def embed_chunk(self, query: str) -> List[float]:
        """
        将单个查询字符串向量化（供记忆查询层调用）
        
        参数:
            query: 用户提示词/查询内容
        
        返回:
            List[float]: 向量表示（已归一化，适合余弦相似度）
        
        示例:
            >>> service = EmbeddingService.get_instance()
            >>> vector = service.embed_chunk("我护照什么时候过期？")
            >>> print(len(vector))  # 512 (BAAI/bge-small-zh-v1.5的维度)
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # 使用model.encode进行向量化
        embedding = self._model.encode(
            [query],  # encode需要列表输入
            batch_size=1,
            normalize_embeddings=True,  # 归一化（余弦相似度空间）
            show_progress_bar=False
        )
        
        # 返回单个向量（List[float]格式）
        return np.asarray(embedding[0], dtype=np.float32).tolist()
    
    def embed_batch(self, queries: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        批量向量化（性能优化）
        
        参数:
            queries: 查询列表
            batch_size: 批处理大小
        
        返回:
            List[List[float]]: 向量列表
        """
        if not queries:
            return []
        
        embeddings = self._model.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(queries) > 64
        )
        
        return np.asarray(embeddings, dtype=np.float32).tolist()


# 便捷函数：供外部快速调用
def embed_chunk(query: str, model_name: str = "BAAI/bge-small-zh-v1.5") -> List[float]:
    """
    便捷函数：将用户查询向量化（供记忆查询层调用）
    
    参数:
        query: 用户提示词内容
        model_name: embedding模型名称（可选）
    
    返回:
        List[float]: query的向量形式（已归一化）
    
    示例:
        >>> from rag_ingest_incremental import embed_chunk
        >>> vector = embed_chunk("我护照什么时候过期？")
        >>> print(f"向量维度: {len(vector)}")
    
    注意:
        - 模型会被缓存，多次调用不会重复加载
        - 返回的向量已归一化，适合余弦相似度计算
    """
    service = EmbeddingService.get_instance(model_name)
    return service.embed_chunk(query)


def load_events_from_jsonl(path: str) -> List[MemoryEvent]:
    events: List[MemoryEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            events.append(MemoryEvent(**obj))
    return events


def write_memory_events(
    events: Iterable[MemoryEvent | Dict[str, Any]],
    *,
    persist_dir: str = ".vector_store",
    embed_model: str = "BAAI/bge-small-zh-v1.5",
    chunk_cfg: Optional[ChunkingConfig] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Tool入口：将 MemoryEvent 列表增量写入向量库。

    - 输入：MemoryEvent 或 dict（会自动转换）
    - 输出：写入结果（records/upserted/errors）
    """
    normalized: List[MemoryEvent] = []
    for e in events:
        if isinstance(e, MemoryEvent):
            normalized.append(e)
        elif isinstance(e, dict):
            normalized.append(MemoryEvent(**e))
        else:
            raise TypeError(f"Unsupported event type: {type(e)}")

    if not normalized:
        return {"records": 0, "upserted": 0, "errors": []}

    cfg = chunk_cfg or ChunkingConfig()
    records = build_chunk_records(normalized, chunk_cfg=cfg)

    service = EmbeddingService.get_instance(embed_model)
    embeddings = service.embed_batch([r.text for r in records], batch_size=batch_size)

    svc = SQLiteVectorStoreService(VectorStoreConfig(persist_dir=persist_dir))
    res = svc.upsert_records(records, embeddings=embeddings)
    return {"records": len(records), **res}


def write_memory_events_from_jsonl(
    path: str,
    *,
    persist_dir: str = ".vector_store",
    embed_model: str = "BAAI/bge-small-zh-v1.5",
    chunk_cfg: Optional[ChunkingConfig] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Tool入口：从 JSONL 读取 MemoryEvent 并写入向量库。
    """
    events = load_events_from_jsonl(path)
    return write_memory_events(
        events,
        persist_dir=persist_dir,
        embed_model=embed_model,
        chunk_cfg=chunk_cfg,
        batch_size=batch_size,
    )


def main() -> None:
    # Windows 终端常见编码导致中文日志乱码；尽量强制 UTF-8 输出方便调试
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--events_jsonl", required=True, help="输入事件 JSONL（每行一个 MemoryEvent）")
    ap.add_argument("--persist_dir", default=".vector_store", help="向量库持久化目录（SQLite 文件会放在这里）")
    ap.add_argument("--collection", default="user_memory_chunks", help="(兼容保留) collection 名称；SQLite实现不使用")
    ap.add_argument("--embed_model", default="BAAI/bge-small-zh-v1.5", help="SentenceTransformer 模型名")
    ap.add_argument("--max_chars", type=int, default=600)
    ap.add_argument("--overlap_chars", type=int, default=80)
    ap.add_argument("--chunk_version", type=int, default=1)
    args = ap.parse_args()

    events = load_events_from_jsonl(args.events_jsonl)
    chunk_cfg = ChunkingConfig(
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        chunk_version=args.chunk_version,
    )
    res = write_memory_events(
        events,
        persist_dir=args.persist_dir,
        embed_model=args.embed_model,
        chunk_cfg=chunk_cfg,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

