from __future__ import annotations

from typing import Any, Dict, Optional

from jcards.api.jcard_service import get_jcard_service
from jcards.core.models import CardWriteOps, Jcard, WriteOpType


def _coerce_jcard(card: Jcard | Dict[str, Any]) -> Jcard:
    if isinstance(card, Jcard):
        return card
    if isinstance(card, dict):
        return Jcard.from_dict(card)
    raise TypeError(f"card must be Jcard or dict, got {type(card)}")


def _build_write_ops(
    action: str,
    *,
    card: Jcard | Dict[str, Any],
    target_card_id: Optional[str] = None,
    expected_version: Optional[int] = None,
    card_expected_version: Optional[int] = None,
    target_expected_version: Optional[int] = None,
) -> CardWriteOps:
    action = action.strip().lower()
    if action == "add":
        op = WriteOpType.UPSERT
    elif action == "correct":
        op = WriteOpType.CORRECT
    elif action == "supersede":
        op = WriteOpType.SUPERSEDE
    elif action == "deactivate":
        op = WriteOpType.DEACTIVATE
    else:
        raise ValueError("action must be one of: Add, Correct, Supersede, Deactivate")

    jcard = _coerce_jcard(card)
    return CardWriteOps(
        op=op,
        card=jcard,
        target_card_id=target_card_id,
        expected_version=expected_version,
        card_expected_version=card_expected_version,
        target_expected_version=target_expected_version,
    )


def update_jcards_database(
    action: str,
    *,
    card: Jcard | Dict[str, Any],
    target_card_id: Optional[str] = None,
    expected_version: Optional[int] = None,
    card_expected_version: Optional[int] = None,
    target_expected_version: Optional[int] = None,
    repository=None,
) -> Dict[str, Any]:
    """
    Tool入口：写入/修正 Jcards 库。

    - action: "Add" | "Correct" | "Supersede" | "Deactivate"
    - card: Jcard 或 dict（必须包含 Jcard.from_dict 需要的字段）
    - target_card_id: 对 Correct / Supersede / Deactivate 生效
    - expected_version/card_expected_version/target_expected_version: 乐观锁字段
    """
    service = get_jcard_service(repository)

    try:
        ops = _build_write_ops(
            action,
            card=card,
            target_card_id=target_card_id,
            expected_version=expected_version,
            card_expected_version=card_expected_version,
            target_expected_version=target_expected_version,
        )
        result = service.apply_card_write_ops(ops)
        return {
            "success": result.applied,
            "upserted_ids": result.upserted_ids,
            "updated_ids": result.updated_ids,
            "superseded_ids": result.superseded_ids,
            "deleted_ids": result.deleted_ids,
            "errors": result.errors,
        }
    except Exception as exc:
        return {
            "success": False,
            "upserted_ids": [],
            "updated_ids": [],
            "superseded_ids": [],
            "deleted_ids": [],
            "errors": [str(exc)],
        }

