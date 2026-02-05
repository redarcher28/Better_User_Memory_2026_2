# tests/conftest.py
"""Pytest fixtures for three-level user memory tests. Ensures project root is on path and provides vector store, Jcards mock, and Agent factory."""
import sys
from pathlib import Path
from typing import List, Optional

# Project root on path so we can import group1, group3, jcards
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest


class MockJcardsDb:
    """Jcards_db stand-in: get_Jcards_tostr() returns a configurable list of card strings."""

    def __init__(self, cards: Optional[List[str]] = None):
        self.cards = list(cards) if cards is not None else []

    def get_Jcards_tostr(self) -> List[str]:
        return self.cards


class MockLLMClient:
    """LLM client that returns predefined responses in order. Used for deterministic tests."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.call_count = 0

    def think(self, messages, temperature=0):
        self.call_count += 1
        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return self.responses[-1] if self.responses else ""


def _make_agent(persist_dir: str, jcards_db, embed_db, llm_client):
    from group1.ToolExecutor import ToolExecutor
    from group1.ReAct import ReActAgent

    tool_executor = ToolExecutor()
    agent = ReActAgent(
        llm_client=llm_client,
        tool_executor=tool_executor,
        jcards_db=jcards_db,
        embed_db=embed_db,
        max_steps=10,
    )
    return agent


@pytest.fixture
def persist_dir(tmp_path):
    """Temporary directory for vector store; isolated per test."""
    d = tmp_path / "vector_store"
    d.mkdir()
    return str(d)


@pytest.fixture
def embed_db(persist_dir):
    """Embed_db using the test vector store directory."""
    pytest.importorskip("sentence_transformers", reason="sentence_transformers required for RAG/embedding")
    from group1.RAG_query import Embed_db
    return Embed_db(persist_dir=persist_dir)


@pytest.fixture
def jcards_db_empty():
    """Jcards with no cards."""
    return MockJcardsDb([])


@pytest.fixture
def agent_factory(persist_dir, embed_db):
    """Factory: (jcards_db, llm_client) -> ReActAgent. Caller supplies jcards and LLM."""

    def factory(jcards_db, llm_client):
        return _make_agent(persist_dir, jcards_db, embed_db, llm_client)

    return factory
