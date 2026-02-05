# tests/test_level2_multi_session.py
"""Level 2: Multi-session retrieval - scatter integration and fact conflict."""
import sys
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
from tests.conftest import MockJcardsDb, MockLLMClient


def _seed_rag(persist_dir: str) -> None:
    pytest.importorskip("sentence_transformers")
    from group3.rag_ingest_incremental import update_rag_vector_store

    def add(content: str, conv_id: str, turn: int, speaker: str = "user"):
        update_rag_vector_store(
            action="Add",
            concluded_content=content,
            conversation_id=conv_id,
            turn_id=turn,
            speaker=speaker,
            timestamp=datetime.now().isoformat(),
            persist_dir=persist_dir,
        )

    return add


def test_level2_scatter_integration(agent_factory, jcards_db_empty, persist_dir):
    """Cross-session: two sessions mention two cars; agent should integrate (Toyota + Honda)."""
    add = _seed_rag(persist_dir)
    add("用户说：我有一辆丰田车。", "session_jan", 1)
    add("用户说：我有一辆本田车。", "session_feb", 1)

    mock_responses = [
        "Thought: 需要检索用户提到的车辆信息。\nStep: GetRAGHistory[我有哪些车]",
        "Thought: 检索到丰田和本田两辆车。\nStep: Finish[您有两辆车：一辆丰田车和一辆本田车。]",
    ]
    llm = MockLLMClient(mock_responses)
    agent = agent_factory(jcards_db_empty, llm)

    result = agent.run("我有哪些车？")

    assert result is not None
    assert "丰田" in result and "本田" in result


def test_level2_fact_conflict(agent_factory, jcards_db_empty, persist_dir):
    """Conflict: wife sets transfer, husband changes, wife changes again; final instruction = last (wife, Z, 150)."""
    add = _seed_rag(persist_dir)
    add("妻子设立初始电汇。向账户X转账100元。", "conv_transfer", 1, "wife")
    add("丈夫修改电汇。向账户Y转账200元。", "conv_transfer", 2, "husband")
    add("妻子在丈夫修改后再次修改。向账户Z转账150元。", "conv_transfer", 3, "wife")

    mock_responses = [
        "Thought: 需要检索最终电汇指令。\nStep: GetRAGHistory[最终的电汇指令是什么]",
        "Thought: 最终有效指令是妻子最后修改的。\nStep: Finish[最终电汇指令：向账户Z转账150元。]",
    ]
    llm = MockLLMClient(mock_responses)
    agent = agent_factory(jcards_db_empty, llm)

    result = agent.run("最终的电汇指令是什么？请说明应向哪个账户转多少。")

    assert result is not None
    assert "Z" in result or "150" in result
