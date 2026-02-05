# tests/test_level3_proactive.py
"""Level 3: Proactive service - agent finds hidden links and gives warnings (e.g. passport + flight)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
from tests.conftest import MockJcardsDb, MockLLMClient


def test_level3_proactive_passport_flight(agent_factory, persist_dir):
    """Jcards: passport expires 2025-02-18, flight to Tokyo 2025-01-15; user asks what to prepare for January Tokyo trip; expect proactive passport reminder."""
    jcards = MockJcardsDb([
        "卡片ID: card_passport\n事实键: passport.expiry\n值: 2025-02-18\n状态: ACTIVE\n置信度: 0.9\n更新时间: 2025-01-01",
        "卡片ID: card_flight\n事实键: flight.tokyo\n值: 2025-01-15\n状态: ACTIVE\n置信度: 0.9\n更新时间: 2025-01-01",
    ])

    mock_responses = [
        "Thought: 查看 Jcards 和检索历史，为一月东京之行做准备。\nStep: GetRAGHistory[一月东京之行 护照 机票]",
        "Thought: 护照2月18日过期，机票1月15日，应主动提醒续签。\nStep: Finish[为一月东京之行，建议您尽快办理护照续签，您的护照将于2025年2月18日过期，建议提前加急续签。]",
    ]
    llm = MockLLMClient(mock_responses)
    agent = agent_factory(jcards, llm)

    result = agent.run("为一月的东京之行，还有什么要准备的吗？")

    assert result is not None
    assert "护照" in result
    assert any(kw in result for kw in ["过期", "续签", "办理"])
