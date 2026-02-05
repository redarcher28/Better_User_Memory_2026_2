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
from group1 import Jcards_db, Embed_db, Active_service
from group1.new import build_system_prompt_with_warning

# 系统提示词模板
AGENT_SYSTEM_PROMPT = """
你是一个有能力调用外部工具的智能助手。每轮对话中，你会看到「当前 Jcards 列表」和「主动警示」；需要检索历史对话片段时请使用 RAG 查询工具，需要增删改聊天记录或 Jcards 时请使用对应的修改工具。

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


class ReActAgent:
    def __init__(
        self,
        llm_client: LLMCompatibleClient,
        tool_executor: ToolExecutor,
        jcards_db: Jcards_db,
        embed_db: Embed_db,
        active_service: Optional[Active_service] = None,
        max_steps: int = 5,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.jcards_db = jcards_db
        self.embed_db = embed_db
        self.active_service = active_service
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
    # route: 1-1 ReAct架构的主循环
    def run(self, question: str):
        self.history = [f"用户请求: {question}"]
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 每轮获取全量 jcards 与主动警示，拼入 prompt
            try:
                jcards_list = self.jcards_db.get_Jcards_tostr()
            except Exception:
                jcards_list = []
            jcards_str = "\n".join(jcards_list) if jcards_list else "（暂无）"
            if self.active_service is not None:
                try:
                    _, active_content = self.active_service.get_active()
                except Exception:
                    active_content = None
            else:
                active_content = None
            active_str = "\n".join(active_content) if active_content else "（无）"

            context_prefix = (
                f"当前 Jcards 列表：\n{jcards_str}\n\n主动警示：\n{active_str}\n\n---\n对话历史：\n"
            )

            # route: 1-1-1 返回系统 prompt（带警示级别拼装）
            base_system_prompt = AGENT_SYSTEM_PROMPT.format(tools=self.tool_executor.getAvailableTools())
            system_prompt = build_system_prompt_with_warning(question, jcards_list, base_system_prompt)
            prompt = context_prefix + "\n".join(self.history)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            # route: 1-1-2 大模型的响应存入 response_text
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("错误：LLM未能返回有效响应。")
                break

            self.history.append(response_text)
            # route: 1-1-3
            thought, action = self._parse_output(response_text)
            if thought:
                print(f"🤔 思考: {thought}")
            else:
                print("警告：未能解析出有效的Action，流程终止。")
                break
            if action is None:
                self.history.append("Observation: 未能解析出 Action，请按格式输出 Action: tool_name[tool_input] 或 Finish[答案]。")
                continue

            # 如果动作类型是 Finish，即模型认为循环可以结束了
            if action.startswith("Finish"):
                # route: 1-1-4
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            # route: 1-1-5
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                self.history.append("Observation: 无效的Action格式，请检查。")
                continue
            tool_input = tool_input.strip()
            if not tool_input:
                self.history.append("Observation: 工具输入不能为空，请提供有效的查询或 JSON。")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            # route: 1-1-6
            tool_function = self.tool_executor.getTool(tool_name)
            # route: 1-1-7 执行 tool_function，参数为 tool_input，返回结果作为 observation
            observation = (
                tool_function(tool_input)
                if tool_function
                else f"错误：未找到名为 '{tool_name}' 的工具。"
            )
            self.history.append(f"Observation: {observation}")
            print(f"👀 观察: {observation}")

        print("已达到最大步数，流程终止。")
        return None
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
    llm = LLMCompatibleClient()
    tool_executor = ToolExecutor()
    jcards_db = Jcards_db()
    embed_db = Embed_db()
    active_service = Active_service()  # 若尚未实现可传 None
    agent = ReActAgent(
        llm_client=llm,
        tool_executor=tool_executor,
        jcards_db=jcards_db,
        embed_db=embed_db,
        active_service=active_service,
    )
    question = "根据历史对话和 Jcards，简要总结与我相关的重要信息；如需检索更多历史请使用 RAG 查询。"
    agent.run(question)
