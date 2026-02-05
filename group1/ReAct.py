import re
import json
from typing import List, Optional

from LLMCompatibleClient import LLMCompatibleClient
from ToolExecutor import (
    ToolExecutor,
    get_rag_history,
    update_rag_vector_store,
    update_jcards_database,
)
from RAG_query import Jcards_db, Embed_db
from new import build_system_prompt_with_warning

# #region agent log
import os
LOG_PATH = r"d:\桌面\Better_User_Memory_2026_2\.cursor\debug.log"
def _log_debug(session_id, run_id, hypothesis_id, location, message, data):
    try:
        import json as _json
        import time
        log_dir = os.path.dirname(LOG_PATH)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps({"sessionId": session_id, "runId": run_id, "hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": time.time() * 1000}, ensure_ascii=False) + "\n")
            f.flush()
    except Exception as e:
        print(f"[DEBUG LOG ERROR] {e}")

# 模块加载时立即写入日志
try:
    _log_debug("debug-session", "run1", "A", "ReAct.py:module_load", "模块开始加载", {})
except:
    pass
# #endregion

# 系统提示词模板
AGENT_SYSTEM_PROMPT = """
你是一个有能力调用外部工具的智能助手。每轮对话中，你会看到「当前 Jcards 列表」；系统已根据当前问题与 Jcards 注入相应级别的安全警示（见本系统提示前文）。需要检索历史对话片段时请使用 RAG 查询工具，需要增删改聊天记录或 Jcards 时请使用对应的修改工具。

写入规则（务必遵守）：
1) **短期、非结构化、对话上下文型信息** → 写入 RAG（UpdateRAG）。例如：临时约定、当天事件、会话中的轶事、位置/时间的即时描述等。
2) **长期稳定、可结构化的个人事实** → 写入 Jcards（UpdateJcards）。例如：姓名、过敏、偏好、固定关系等。
3) **闲聊/情绪/无信息增量** → 不写入任何库。
4) 当内容更偏“上下文细节”而非“稳定事实”时，优先写入 RAG。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `tool_name[tool_input]`: 调用一个可用工具。其中 GetRAGHistory 的 tool_input 为查询字符串；UpdateRAG 与 UpdateJcards 的 tool_input 为 JSON 字符串（见上方工具说明）。
- `Finish[最终答案]`: 当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在 Action: 字段后使用 `Finish["..."]` 来输出最终答案。

现在，请开始吧！
"""

# 用户输入以下命令之一时结束交互式会话
EXIT_COMMANDS = ("quit", "exit", "再见")


class ReActAgent:
    def __init__(
        self,
        llm_client: LLMCompatibleClient,
        tool_executor: ToolExecutor,
        jcards_db: Jcards_db,
        embed_db: Embed_db,
        max_steps: int = 5,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.jcards_db = jcards_db
        self.embed_db = embed_db
        self.max_steps = max_steps
        self.history: List[str] = []

        # 注册三个包装后的工具（单字符串入参，供 ReAct Action 调用）
        tool_executor.registerTool(
            "GetRAGHistory",
            "从 RAG 中检索与查询相关的历史对话片段。输入为查询字符串（当前问题或关键词）。",
            self._wrap_get_rag_history,
        )
        tool_executor.registerTool(
            "UpdateRAG",
            "添加或修改 RAG 向量库中的聊天记录。输入为 JSON 字符串，包含 action（Add/Correct）、concluded_content；Add 时需 conversation_id、turn_id、speaker、timestamp；Correct 时需 chunk_ids，可选 correct_behavior（replace/overwrite）。",
            self._wrap_update_rag,
        )
        tool_executor.registerTool(
            "UpdateJcards",
            "添加、修改或删除 Jcards 库中的卡片。输入为 JSON：action（Add/Correct/Delete）；card_content 为结构化对象（title、body 必填，tags、metadata 可选），Add/Correct 时必填；card_ids 为卡片稳定 ID 列表，Correct/Delete 时必填。返回 added_ids/updated_ids/deleted_ids/errors 供判断成功与否。",
            self._wrap_update_jcards,
        )

    def _wrap_get_rag_history(self, tool_input: str) -> str:
        """包装 get_rag_history：tool_input 即 query，返回片段拼接成的字符串。"""
        try:
            chunks = get_rag_history(
                query=tool_input.strip(),
                jcards_db=self.jcards_db,
                embed_db=self.embed_db,
            )
            if isinstance(chunks, list):
                return "\n".join(chunks) if chunks else "（未检索到相关历史片段。）"
            return str(chunks)
        except Exception as e:
            return f"RAG 查询出错: {e}"

    def _wrap_update_rag(self, tool_input: str) -> str:
        """包装 update_rag_vector_store：tool_input 为 JSON，解析后调用并返回结果描述。"""
        try:
            data = json.loads(tool_input.strip())
            action = data.get("action")
            concluded_content = data.get("concluded_content", "")
            chunk_ids = data.get("chunk_ids")
            conversation_id = data.get("conversation_id")
            turn_id = data.get("turn_id")
            speaker = data.get("speaker")
            timestamp = data.get("timestamp")
            correct_behavior = data.get("correct_behavior", "replace")
            result = update_rag_vector_store(
                action=action,
                concluded_content=concluded_content,
                chunk_ids=chunk_ids,
                conversation_id=conversation_id,
                turn_id=turn_id,
                speaker=speaker,
                timestamp=timestamp,
                correct_behavior=correct_behavior,
            )
            if result is None:
                return "RAG 向量库已更新（具体实现待后续补齐）。"
            upserted, updated, deleted, errors = result
            parts = []
            if upserted:
                parts.append(f"upserted_ids: {upserted}")
            if updated:
                parts.append(f"updated_ids: {updated}")
            if deleted:
                parts.append(f"deleted_ids: {deleted}")
            if errors:
                parts.append(f"errors: {errors}")
            return "; ".join(parts) if parts else "RAG 向量库已更新。"
        except json.JSONDecodeError as e:
            return f"UpdateRAG 输入不是合法 JSON: {e}"
        except Exception as e:
            return f"UpdateRAG 执行出错: {e}"

    def _wrap_update_jcards(self, tool_input: str) -> str:
        """包装 update_jcards_database：tool_input 为 JSON，解析后调用并返回 added_ids/updated_ids/deleted_ids/errors。"""
        try:
            data = json.loads(tool_input.strip())
            action = data.get("action")
            card_content = data.get("card_content")
            card_ids = data.get("card_ids")
            result = update_jcards_database(
                action=action, card_content=card_content, card_ids=card_ids
            )
            if result is None:
                return "Jcards 已更新（具体实现待后续补齐）。"
            added_ids, updated_ids, deleted_ids, errors = result
            parts = []
            if added_ids:
                parts.append(f"added_ids: {added_ids}")
            if updated_ids:
                parts.append(f"updated_ids: {updated_ids}")
            if deleted_ids:
                parts.append(f"deleted_ids: {deleted_ids}")
            if errors:
                parts.append(f"errors: {errors}")
            return "; ".join(parts) if parts else "Jcards 已更新。"
        except json.JSONDecodeError as e:
            return f"UpdateJcards 输入不是合法 JSON: {e}"
        except Exception as e:
            return f"UpdateJcards 执行出错: {e}"
    def _process_single_turn(
        self, question: str, history_prefix: Optional[List[str]] = None
    ) -> Optional[str]:
        """执行单次推理循环：对当前问题运行 ReAct 步骤直至 Finish 或达到最大步数。
        若提供 history_prefix，会拼在当前轮之前，用于多轮对话上下文。
        返回最终答案字符串，或 None（未得到答案/出错/达最大步数）。
        """
        # #region agent log
        _log_debug("debug-session", "run1", "B", "ReAct.py:155", "_process_single_turn() 开始", {"question": question, "max_steps": self.max_steps, "has_history_prefix": history_prefix is not None})
        # #endregion
        if history_prefix is not None:
            self.history = list(history_prefix) + [f"用户请求: {question}"]
        else:
            self.history = [f"用户请求: {question}"]
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")
            # #region agent log
            _log_debug("debug-session", "run1", "B", "ReAct.py:167", "进入循环步骤", {"current_step": current_step, "max_steps": self.max_steps})
            # #endregion

            try:
                jcards_list = self.jcards_db.get_Jcards_tostr()
            except Exception:
                jcards_list = []
            jcards_str = "\n".join(jcards_list) if jcards_list else "（暂无）"
            context_prefix = (
                f"当前 Jcards 列表：\n{jcards_str}\n\n---\n对话历史：\n"
            )

            base_system_prompt = AGENT_SYSTEM_PROMPT.format(
                tools=self.tool_executor.getAvailableTools()
            )
            system_prompt = build_system_prompt_with_warning(
                question, jcards_list, base_system_prompt
            )
            prompt = context_prefix + "\n".join(self.history)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            # #region agent log
            _log_debug("debug-session", "run1", "C", "ReAct.py:200", "调用 LLM 前", {"messages_count": len(messages)})
            # #endregion
            response_text = self.llm_client.think(messages=messages)
            # #region agent log
            _log_debug("debug-session", "run1", "C", "ReAct.py:201", "LLM 调用后", {"response_text": response_text[:200] if response_text else None, "is_empty": not response_text, "is_none": response_text is None})
            # #endregion
            if not response_text:
                print("错误：LLM未能返回有效响应。")
                # #region agent log
                _log_debug("debug-session", "run1", "C", "ReAct.py:203", "LLM 返回空，提前 break", {})
                # #endregion
                break

            self.history.append(response_text)
            thought, action = self._parse_output(response_text)
            # #region agent log
            _log_debug("debug-session", "run1", "D", "ReAct.py:206", "解析输出后", {"thought": thought[:100] if thought else None, "action": action, "has_thought": thought is not None, "has_action": action is not None})
            # #endregion
            if thought:
                print(f"🤔 思考: {thought}")
            else:
                print("警告：未能解析出有效的Action，流程终止。")
                # #region agent log
                _log_debug("debug-session", "run1", "D", "ReAct.py:210", "没有解析出 thought，提前 break", {})
                # #endregion
                break
            if action is None:
                self.history.append(
                    "Observation: 未能解析出 Action，请按格式输出 Action: tool_name[tool_input] 或 Finish[答案]。"
                )
                # #region agent log
                _log_debug("debug-session", "run1", "D", "ReAct.py:213", "action 为 None，继续循环", {})
                # #endregion
                continue

            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案: {final_answer}")
                # #region agent log
                _log_debug("debug-session", "run1", "F", "ReAct.py:219", "检测到 Finish action，准备返回", {"final_answer": final_answer})
                # #endregion
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                self.history.append("Observation: 无效的Action格式，请检查。")
                continue
            tool_input = tool_input.strip()
            if not tool_input:
                self.history.append(
                    "Observation: 工具输入不能为空，请提供有效的查询或 JSON。"
                )
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            tool_function = self.tool_executor.getTool(tool_name)
            observation = (
                tool_function(tool_input)
                if tool_function
                else f"错误：未找到名为 '{tool_name}' 的工具。"
            )
            self.history.append(f"Observation: {observation}")
            print(f"👀 观察: {observation}")

        print("已达到最大步数，流程终止。")
        # #region agent log
        _log_debug("debug-session", "run1", "B", "ReAct.py:245", "达到最大步数，返回 None", {"current_step": current_step, "max_steps": self.max_steps})
        # #endregion
        return None

    # route: 1-1 ReAct架构的主循环（单次调用入口）
    def run(self, question: str) -> Optional[str]:
        """单次对话：对一个问题执行 ReAct 并返回最终答案。"""
        # #region agent log
        _log_debug("debug-session", "run1", "A", "ReAct.py:250", "run() 方法被调用", {"question": question})
        # #endregion
        result = self._process_single_turn(question)
        # #region agent log
        _log_debug("debug-session", "run1", "A", "ReAct.py:252", "run() 方法返回", {"result": result, "result_type": type(result).__name__, "is_none": result is None})
        # #endregion
        return result

    def start_interactive_session(self) -> None:
        """启动交互式会话：支持用户多次输入，由用户输入退出命令结束对话。
        每轮会复用上一轮的对话历史，便于多轮上下文。
        """
        print("\n=== ReAct Agent 交互式会话 ===")
        print("输入 'quit'、'exit' 或 '再见' 结束对话。\n")

        session_history: List[str] = []

        while True:
            try:
                user_input = input("👤 您：").strip()
            except EOFError:
                print("\n🤖 Agent：再见！感谢与您的对话。")
                break

            if user_input.lower() in EXIT_COMMANDS:
                print("🤖 Agent：再见！感谢与您的对话。")
                break

            if not user_input:
                continue

            result = self._process_single_turn(user_input, history_prefix=session_history)
            session_history = list(self.history)
            if result is not None:
                print(f"\n🤖 Agent：{result}\n")
            else:
                print("\n🤖 Agent：（本轮未能得到答案，您可以继续提问。）\n")
    # route: 1-1-3 将模型的thought和action从模型输出text中分离出来，返回thought, action
    def _parse_output(self, text: str):
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    # route: 1-1-5
    #  输入示例：
    #  action_text = "Search[OpenAI最新消息]"
    #  _parse_action 处理后：
    #  返回("Search", "OpenAI最新消息")
    def _parse_action(self, action_text: str):
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        return (match.group(1), match.group(2)) if match else (None, None)

    # route: 1-1-4
    #  用户问："中国的首都是哪里？"
    #  Agent思考：
    #  1.我需要找到中国的首都
    #  2.我知道是北京
    #  3.我应该输出Finish[北京]
    #  该函数功能为提取Finish后【】里的字符串
    def _parse_action_input(self, action_text: str):
        match = re.match(r"Finish\[(.*)\]", action_text, re.DOTALL)
        # match = re.match(r"\w+\[(.*)\]", action_text)
        return match.group(1) if match else ""


if __name__ == "__main__":
    print("ReAct.py 模块加载完成")
    import sys
    import traceback
    try:
        print("[DEBUG] 程序开始执行...")
        # #region agent log
        _log_debug("debug-session", "run1", "A", "ReAct.py:360", "程序开始执行", {"argv": sys.argv})
        print(f"[DEBUG] 日志已写入: {LOG_PATH}")
        # #endregion
        try:
            llm = LLMCompatibleClient(
                model="deepseek-chat",
                apiKey="sk-55950ea43bc44fb58e5379fc9f2c1d2a",
                baseUrl="https://api.deepseek.com",
                timeout=60
            )
            # #region agent log
            _log_debug("debug-session", "run1", "A", "ReAct.py:363", "LLMCompatibleClient 初始化成功", {})
            # #endregion
        except Exception as e:
            # #region agent log
            _log_debug("debug-session", "run1", "A", "ReAct.py:366", "LLMCompatibleClient 初始化失败", {"error": str(e), "traceback": traceback.format_exc()})
            # #endregion
            print(f"[ERROR] LLMCompatibleClient 初始化失败: {e}")
            raise
        tool_executor = ToolExecutor()
        jcards_db = Jcards_db()
        embed_db = Embed_db()
        # #region agent log
        _log_debug("debug-session", "run1", "A", "ReAct.py:373", "所有组件初始化完成，创建 ReActAgent", {})
        # #endregion
        agent = ReActAgent(
            llm_client=llm,
            tool_executor=tool_executor,
            jcards_db=jcards_db,
            embed_db=embed_db,
        )
        # #region agent log
        _log_debug("debug-session", "run1", "A", "ReAct.py:381", "ReActAgent 创建成功", {})
        # #endregion

        # 默认进入交互式会话（用户问答环节）；传 --single-run 时只执行一次默认问题后退出
        if len(sys.argv) > 1 and sys.argv[1] == "--single-run":
            question = "根据历史对话和 Jcards，简要总结与我相关的重要信息；如需检索更多历史请使用 RAG 查询。"
            # #region agent log
            _log_debug("debug-session", "run1", "A", "ReAct.py:329", "主程序调用 run()", {"question": question})
            # #endregion
            print(f"[DEBUG] 准备调用 agent.run(question)...")
            result = agent.run(question)
            print(f"[DEBUG] agent.run() 返回: {result} (类型: {type(result)})")
            # #region agent log
            _log_debug("debug-session", "run1", "A", "ReAct.py:331", "主程序收到 run() 返回值", {"result": result, "result_type": type(result).__name__, "is_none": result is None, "will_print": False})
            # #endregion
            if result is not None:
                print(f"\n🤖 Agent 最终答案: {result}")
            else:
                print("\n⚠️ Agent 未能返回答案")
            print("[DEBUG] 程序执行完成")
        else:
            agent.start_interactive_session()
    except Exception as e:
        error_msg = f"[FATAL ERROR] 程序执行出错: {e}\n{traceback.format_exc()}"
        print(error_msg)
        # #region agent log
        try:
            _log_debug("debug-session", "run1", "A", "ReAct.py:main_except", "程序执行出错", {"error": str(e), "traceback": traceback.format_exc()})
        except:
            pass
        # #endregion
        sys.exit(1)
