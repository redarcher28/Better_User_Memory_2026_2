from typing import List
import re


# 高风险关键词库（可根据业务扩展）
HIGH_RISK_KEYWORDS = {
    'medical': ['诊断', '处方', '手术', '药物', '过敏', '癌症', '治疗'],
    'finance': ['转账', '贷款', '账户', '密码', '交易', '冻结', '信用'],
    'legal': ['合同', '诉讼', '违法', '责任', '赔偿', '律师'],
    'safety': ['危险', '爆炸', '泄漏', '紧急', '事故', '死亡']
}


def _contains_risk_keywords(text: str) -> float:
    """基于关键词匹配返回风险分数（0.0～0.6）"""
    text_lower = text.lower()
    risk_score = 0.0
    for category, keywords in HIGH_RISK_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            risk_score += min(0.2 * matches, 0.2)
    return min(risk_score, 0.6)


def _assess_jcards_coverage(query: str, jcards: List[str]) -> float:
    """
    简单评估 Jcards 对 query 的覆盖程度。
    返回：覆盖越差，分数越高（0.0=完全覆盖，0.4=完全无关）
    """
    if not jcards:
        return 0.4
    query_words = set(re.findall(r'\w+', query.lower()))
    all_jcards_text = " ".join(jcards).lower()
    matched = sum(1 for w in query_words if w in all_jcards_text)
    coverage_ratio = matched / max(len(query_words), 1)
    return 0.4 * (1 - coverage_ratio)


def _detect_conflict_signals(query: str, jcards: List[str]) -> float:
    """
    检测潜在冲突信号（如否定词+矛盾陈述）。
    返回 0.0～0.3
    """
    conflict_triggers = ['不', '错误', '不对', '纠正', '修改', '更新', '过时']
    if any(trigger in query for trigger in conflict_triggers) and jcards:
        return 0.3
    return 0.0


def estimate_severity(query: str, jcards: List[str]) -> float:
    """
    自动估算警示严重程度 [0.0, 1.0]。

    Args:
        query: 用户输入
        jcards: 参考信息列表

    Returns:
        float: severity 值
    """
    risk_score = _contains_risk_keywords(query)
    coverage_penalty = _assess_jcards_coverage(query, jcards)
    conflict_score = _detect_conflict_signals(query, jcards)
    severity = min(
        risk_score * 0.5 + coverage_penalty * 0.3 + conflict_score * 0.2,
        1.0
    )
    severity = max(severity, 0.05)
    return round(severity, 2)


def get_warning_prompt(severity: float) -> str:
    """
    根据警示严重程度返回对应的系统提示词。

    Args:
        severity: 警示严重程度，范围 [0, 1]

    Returns:
        str: 对应级别的提示词
    """
    if not (0.0 <= severity <= 1.0):
        raise ValueError("Severity must be between 0.0 and 1.0")

    if severity < 0.3:
        return (
            "你是一个智能助手，请基于以下提供的参考信息（Jcards）回答用户问题。\n"
            "若参考信息与问题相关但不完整，可适当补充常识；若完全无关，请说明无法回答。\n"
            "回答应保持礼貌、简洁，并避免猜测。"
        )
    elif severity < 0.6:
        return (
            "【中度警示】你必须严格依据以下参考信息（Jcards）回答问题。\n"
            "仅当 Jcards 明确包含相关信息时才作答；若信息模糊、缺失或存在歧义，请明确回复：\n"
            "\"根据当前信息无法确定，请提供更具体的细节。\"\n"
            "禁止自行推断、编造或假设未提及的内容。"
        )
    else:
        return (
            "【高度警示｜高风险场景】你只能使用以下参考信息（Jcards）中的内容进行回答。\n"
            "任何超出 Jcards 范围的回答都可能导致严重后果。\n"
            "如果 Jcards 未明确、直接支持某个结论，必须回复：\n"
            "\"出于安全考虑，当前信息不足以支持回答该问题。\"\n"
            "不得解释、推测、举例或尝试帮助性补充。"
        )


def build_system_prompt_with_warning(
    question: str, jcards_list: List[str], base_system_prompt: str
) -> str:
    """
    根据当前问题与 jcards 估算警示级别，拼装「警示系统提示词 + 基础系统提示词」。

    不调用任何 LLM/API，仅做计算与字符串拼装。

    Args:
        question: 当前用户问题
        jcards_list: 当前 Jcards 列表（字符串列表）
        base_system_prompt: 基础系统提示词（如 ReAct 的 AGENT_SYSTEM_PROMPT.format(tools=...)）

    Returns:
        str: 完整系统提示词（警示前缀 + 空行 + base_system_prompt）
    """
    severity = estimate_severity(question, jcards_list)
    warning = get_warning_prompt(severity)
    return warning + "\n\n" + base_system_prompt


if __name__ == "__main__":
    # 仅测试警示与拼装逻辑，不调用 API
    sample_jcards = ["示例Jcard内容：今日天气晴朗，温度25°C."]
    q = "今天天气如何？"
    sev = estimate_severity(q, sample_jcards)
    print(f"severity: {sev}")
    print(f"warning prefix: {get_warning_prompt(sev)[:80]}...")
    base = "可用工具如下:\n{tools}\n请严格按照以下格式回应."
    full = build_system_prompt_with_warning(q, sample_jcards, base)
    print(f"build_system_prompt_with_warning 前 100 字符: {full[:100]}...")
