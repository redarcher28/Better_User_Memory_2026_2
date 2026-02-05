# tests/test_level1_basic_recall.py
"""Level 1: Basic recall - agent accurately recalls specific information from a single session."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
from datetime import datetime

from tests.conftest import MockJcardsDb, MockLLMClient


def seed_rag_checking_account(persist_dir: str) -> None:
    """Write one chunk to RAG: user said checking account number is 12345678."""
    pytest.importorskip("sentence_transformers")
    from group3.rag_ingest_incremental import update_rag_vector_store

    update_rag_vector_store(
        action="Add",
        concluded_content="用户说：我的支票账户号码是 12345678。",
        conversation_id="conv_l1_test",
        turn_id=1,
        speaker="user",
        timestamp=datetime.now().isoformat(),
        persist_dir=persist_dir,
    )


@pytest.mark.parametrize("use_mock_llm", [True], ids=["mock_llm"])
def test_level1_basic_recall_mock(agent_factory, jcards_db_empty, persist_dir, use_mock_llm):
    """With pre-seeded RAG and mock LLM, agent returns the stored account number."""
    seed_rag_checking_account(persist_dir)

    mock_responses = [
        "Thought: 用户问支票账户号码，需要从历史中检索。\nStep: GetRAGHistory[我的支票账户号码是多少]",
        "Thought: 检索到内容为：用户说：我的支票账户号码是 12345678。\nStep: Finish[您的支票账户号码是 12345678。]",
    ]
    llm = MockLLMClient(mock_responses)
    agent = agent_factory(jcards_db_empty, llm)

    result = agent.run("我的支票账户号码是多少？")

    assert result is not None
    assert "12345678" in result
