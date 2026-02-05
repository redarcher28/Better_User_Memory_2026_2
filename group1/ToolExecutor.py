from dotenv import load_dotenv
import os
from serpapi import SerpApiClient
from typing import Dict, Any, List, TypedDict
from datetime import datetime
import pytz

import sympy as sp
import numpy as np
from scipy import optimize

# 加载 .env 文件中的环境变量
load_dotenv()

# 引用同目录下 group1 模块中的 RAG 查询与数据库接口（与 group1.py 同目录，直接 import group1）
try:
    from . import group1 as _group1
except ImportError:
    import group1 as _group1


def get_rag_history(
    query: str,
    jcards_db: _group1.Jcards_db,
    embed_db: _group1.Embed_db,
) -> List[str]:
    """
    引用RAG，返回以前需要的对话记录的片段。
    内部调用 group1.RAG_query.return_reranked_chunks 完成检索与重排。

    详细说明：
    - 此函数用于从RAG系统中检索以前的对话记录片段
    - 这些片段是系统认为与当前任务相关的历史对话内容
    - 返回的是一个字符串列表，每个字符串代表一个对话记录片段

    参数：
    - query: 用户当前查询/提示词，用于检索相关片段
    - jcards_db: Jcards 数据库实例
    - embed_db: 向量数据库实例

    返回值：
    - List[str]: 包含与 query 最相关的对话记录片段列表（重排后的 top 片段）
    """
    rag = _group1.RAG_query()
    return rag.return_reranked_chunks(query=query, jcards_db=jcards_db, embed_db=embed_db)


def update_rag_vector_store(
    action: str,
    concluded_content: str,
    *,
    chunk_ids: List[str] | None = None,
    conversation_id: str | None = None,
    turn_id: int | None = None,
    speaker: str | None = None,
    timestamp: str | None = None,
    correct_behavior: str = "replace",
) -> tuple[List[str], List[str], List[str], List[str]]:
    """
    para:
    action: str, 具体的操作类型，有:
    {
    "Add" : 添加新的聊天记录。此时必须提供 conversation_id / turn_id / speaker / timestamp；
            chunk_ids 为 None。
    "Correct" : 修改错误的聊天记录。此时必须提供 chunk_ids（要修改的 chunk 的可追溯标识）；
                concluded_content 为替换后的内容。
    }
    concluded_content: str, 模型从提示词和自己生成的内容中总结出的聊天记录（Add 为新增内容，Correct 为替换内容）。
    chunk_ids: List[str] | None, 可追溯的 chunk 标识列表。实现上可为 chunk_id，或由
               (conversation_id, turn_range, chunk_version) 生成的稳定 ID。Correct 时必填，Add 时为 None。
    conversation_id: str | None, 会话 ID，Add 时必填，用于与现有切分/追溯逻辑一致。
    turn_id: int | None, 轮次 ID，Add 时必填。
    speaker: str | None, 发言者（如 "user" / "assistant"），Add 时必填。
    timestamp: str | None, 时间戳，Add 时必填。
    correct_behavior: str, 仅 Correct 时有效。可选:
    {
    "overwrite" : 覆盖同 id：在原 chunk_id 上原地更新内容，旧内容不再被检索。
    "replace"   : 逻辑删除旧 chunk + 写入新 chunk：旧 chunk 标记删除不再命中，新 chunk 使用新 id 或版本。
                  默认 "replace"，避免旧内容继续被检索命中。
    }

    返回值：tuple[List[str], List[str], List[str], List[str]]
    元组 (upserted_ids, updated_ids, deleted_ids, errors)，供 Agent 判断是否执行成功。

    详细说明： 此函数用于 Agent 添加或修改 RAG 向量库的内容；写入时需至少包含
    conversation_id / turn_id / speaker / timestamp 等元数据，否则无法按现有切分与追溯逻辑构建 chunk。

    - 添加内容（Add）：
    在片段库和向量库中追加新 chunk。必须提供 conversation_id、turn_id、speaker、timestamp，
    与 concluded_content 一起构成可追溯的增量；仅有 concluded_content 不足以做可靠增量。

    - 修改内容（Correct）：
    根据 chunk_ids 精确定位要修改的 chunk（chunk_id 或由 conversation_id + turn_range + chunk_version 确定）。
    行为由 correct_behavior 决定：overwrite 原地覆盖同 id；replace 逻辑删除旧 chunk 再写新 chunk，
    避免旧内容继续被检索命中。

    - 总结式写入（source="summary"）：
    若为独立总结 chunk，则作为新 chunk 写入并在 metadata 中标记 source="summary"；
    若为替换原事件，则与 Correct 语义配合，并在 metadata 中标识 source="summary"。

    - 修改方式（实现参考）：
      1. 根据 chunk_ids 或元数据生成/解析可追溯标识，保证 Correct 可精确修改。
      2. 写入时附带 conversation_id、turn_id、speaker、timestamp、source 等元数据。
      3. Correct 时按 correct_behavior 执行覆盖或逻辑删除+插入。
      4. 返回元组 (upserted_ids, updated_ids, deleted_ids, errors)，供 Agent 判断是否执行成功。
    """
    # 函数体暂时为空，等待后续实现具体的向量库修改逻辑
    pass


class CardContent(TypedDict, total=False):
    """卡片内容结构化对象。title、body 为 Add/Correct 必填；tags、metadata 可选。"""
    title: str
    body: str
    tags: List[str]
    metadata: Dict[str, Any]


def update_jcards_database(
    action: str,
    card_content: Dict[str, Any] | None,
    card_ids: List[str] | None,
) -> tuple[List[str], List[str], List[str], List[str]]:
    """
    对 Jcards 库执行添加、修改或删除；使用稳定可追溯的 card_id，避免删错/改错。

    参数
    -----
    action: str
        "Add"：添加新卡片，此时 card_ids 为 None，card_content 必填。
        "Correct"：修改已有卡片，card_ids 为要修改的卡片稳定 ID 列表，card_content 为替换后的内容。
        "Delete"：删除卡片，card_ids 为要删除的卡片稳定 ID 列表，card_content 为 None。
    card_content: dict | None
        结构化卡片内容，Add/Correct 时必填，Delete 时为 None。建议结构：
        - title (str)：标题，必填
        - body (str)：正文，必填
        - tags (list[str])：标签列表，可选
        - metadata (dict)：扩展元数据（如 external_id 用于去重），可选
        库内会做标准化与索引，调用方无需关心实现细节。
    card_ids: List[str] | None
        要操作（Correct/Delete）的卡片稳定 ID 列表，可追溯、不随排序变化。Add 时为 None；Correct/Delete 时必填。

    返回值
    -----
    tuple[List[str], List[str], List[str], List[str]]
        (added_ids, updated_ids, deleted_ids, errors)，供 Agent 判断成功/失败。
        - added_ids：本次 Add 成功写入的卡片 ID 列表。
        - updated_ids：本次 Correct 成功更新的卡片 ID 列表。
        - deleted_ids：本次 Delete 成功删除的卡片 ID 列表。
        - errors：错误信息列表（如某 card_id 不存在、重复添加等），每项建议包含 card_id 与原因。

    行为说明
    -----
    - Correct/Delete 幂等性：某 card_id 不存在时，该 id 会出现在 errors 中（如 "card_id:xxx not_found"），其它 id 照常执行；重复调用已删除或已更新的 id 可得到一致结果。
    - Add 去重：是否允许重复由实现决定。若按 content_hash 或 metadata.external_id 判重且不允许重复，重复添加时应在 errors 中返回相应说明，调用方可根据返回值判断。
    """
    # 函数体暂时为空，等待后续实现具体的 Jcards 库修改逻辑；返回空结果以保持接口一致
    return ([], [], [], [])


#
#
# def search(query: str) -> str:
#     """
#     一个基于SerpApi的实战网页搜索引擎工具。
#     它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
#     """
#     print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
#     try:
#         api_key = os.getenv("SERPAPI_API_KEY")
#         if not api_key:
#             return "错误：SERPAPI_API_KEY 未在 .env 文件中配置。"
#
#         params = {
#             "engine": "google",
#             "q": query,
#             "api_key": api_key,
#             "gl": "cn",  # 国家代码
#             "hl": "zh-cn",  # 语言代码
#         }
#
#         client = SerpApiClient(params)
#         results = client.get_dict()
#
#         # 智能解析：优先寻找最直接的答案
#         if "answer_box_list" in results:
#             return "\n".join(results["answer_box_list"])
#         if "answer_box" in results and "answer" in results["answer_box"]:
#             return results["answer_box"]["answer"]
#         if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
#             return results["knowledge_graph"]["description"]
#         if "organic_results" in results and results["organic_results"]:
#             # 如果没有直接答案，则返回前三个有机结果的摘要
#             snippets = [
#                 f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
#                 for i, res in enumerate(results["organic_results"][:3])
#             ]
#             return "\n\n".join(snippets)
#
#         return f"对不起，没有找到关于 '{query}' 的信息。"
#
#     except Exception as e:
#         return f"搜索时发生错误: {e}"
#
# # 时间工具
# # route 1-1-7
# def get_current_time(timezone: str = "Asia/Shanghai") -> str:
#     """
#     一个获取指定时区当前时间的工具。
#     默认返回中国标准时间（北京时间）。
#     参数:
#         timezone: 时区字符串，如 "Asia/Shanghai", "America/New_York", "UTC" 等
#     返回:
#         格式化的当前时间字符串，或错误信息
#     """
#     print(f"⏰ 正在获取 {timezone} 的当前时间...")
#     try:
#         # 获取时区对象
#         tz = pytz.timezone(timezone)
#         # 获取当前时间并转换为指定时区
#         current_time = datetime.now(tz)
#         # 格式化输出
#         formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
#
#         return f"当前 {timezone} 时间: {formatted_time}"
#
#     except pytz.exceptions.UnknownTimeZoneError:
#         return f"错误：未知的时区 '{timezone}'。请使用如 'Asia/Shanghai' 这样的有效时区标识符。"
#     except Exception as e:
#         return f"获取时间时发生错误: {e}"
#
# # # 代码执行工具
# # def codeInterpreter(code: str):
# #     try:
# #         local_vars = {}
# #         exec(code, {"sp": sp, "np": np, "optimize": optimize}, local_vars)
# #         return local_vars.get("result", None)
# #     except Exception as e:
# #         return f"EXECUTION ERROR: {str(e)}"

class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        # route: 1-1-1 引用参数：可引用的工具函数字典集，格式如下：

        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告：工具 '{name}' 已存在，将被覆盖。")

        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    # route: 1-1-6 根据名称获取一个工具的执行函数, name: 工具名称， 返回工具函数
    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        #Tools的数据类型： Dict[str, Dict[str, Any]]
        # 例： self.tools = {
        #             "search": {
        #                 "name": "search",
        #                 "description": "在网络上搜索信息",
        #                 "func": self.search_web  # ← 存储函数引用
        #             },
        #             "calculate": {
        #                 "name": "calculate",
        #                 "description": "执行数学计算",
        #                 "func": self.calculate_expression  # ← 存储函数引用
        #             }
        # }
        # name为工具名称，func存储函数引用
        return self.tools.get(name, {}).get("func")
    # route: 1-1-1 获取所有可用工具的格式化描述字符串。
    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])


# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)
    time_description = "一个获取当前时间的工具。当你需要回答关于时事的问题时，应使用此工具获取最新的时间。"
    toolExecutor.registerTool("Time", time_description, get_current_time)

    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误：未找到名为 '{tool_name}' 的工具。")