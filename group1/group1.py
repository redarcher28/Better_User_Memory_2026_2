from typing import List, Tuple, Optional, Any
import json


class Jcards_db:
    """Jcards数据库接口类"""

    def get_Jcards_tostr(self) -> List[str]:
        """
        从jcards库中获取所有jcards，返回字符串列表形式
        """
        pass


class Embed_db:
    """向量数据库接口类"""

    def load_events_from_json(self, content_in_json: json) -> None:
        """
        将模型返回的内容的json形式给向量库
        """
        pass

    def query(self, query_embeddings: List[float], top_k: int) -> List[str]:
        """
        根据查询向量召回最相关的top_k个片段

        Args:
            query_embeddings: 用户提示词的向量形式
            top_k: 返回前top_k个召回分数最高的片段

        Returns:
            召回片段内容列表
        """
        pass


class Active_service:
    """主动服务接口类"""

    def get_active(self) -> Tuple[bool, Optional[List[str]]]:
        """
        返回是否主动服务，即布尔值Active，和主动服务卡内容，即active_content。
        如果Active为False，则active_content为none

        Returns:
            Tuple[bool, Optional[List[str]]]: (是否主动服务, 主动服务卡内容)
        """
        pass


class RAG_query:
    """
    RAG查询类，负责处理用户查询的召回和重排
    """

    def __init__(self):
        """初始化RAG查询类"""
        self.Jcards: List[str] = []  # 从jcards库中获取的jcards，原先为json格式
        self.shuffled_Jcards: List[str] = []  # 根据query筛选后的Jcards
        self.query: str = ""  # 用户提示词
        self.top_k: int = 3  # 前k个分值最高的召回片段，默认3
        self.retrieved_chunks: List[str] = []  # retrieve的返回值，是召回的片段内容
        self.reranked_chunks: List[str] = []  # rerank返回的值

    def ask_Jcards(self, jcards_db: Jcards_db) -> None:
        """
        将jcards库中拿到所有jcards，赋值给属性Jcards

        Args:
            jcards_db: Jcards数据库实例
        """
        self.Jcards = jcards_db.get_Jcards_tostr()

    def shuffle_Jcards(self, query: str) -> None:
        """
        将所有Jcards中筛选与提示词相关的Jcards，存入属性shuffled_Jcards

        Args:
            query: 用户查询
        """
        if not self.Jcards:
            self.shuffled_Jcards = []
            return

        # 这里实现筛选逻辑，可以根据query与Jcards的相关性进行筛选
        # 示例：简单的关键词匹配筛选
        relevant_jcards = []
        query_keywords = query.lower().split()

        for jcard in self.Jcards:
            jcard_lower = jcard.lower()
            # 检查是否有任何关键词出现在jcard中
            if any(keyword in jcard_lower for keyword in query_keywords):
                relevant_jcards.append(jcard)

        self.shuffled_Jcards = relevant_jcards if relevant_jcards else self.Jcards

    def retrieve(self, query: str, embed_db: Embed_db, top_k: int = None) -> None:
        """
        根据用户提示词和Jcards召回，赋值召回的片段内容给retrieved_chunks

        Args:
            query: 用户查询
            embed_db: 向量数据库实例
            top_k: 召回的数量，如果为None则使用类属性中的top_k
        """
        if top_k is not None:
            self.top_k = top_k

        # 使用Jcards作为上下文进行召回
        # 首先需要将query转换为向量
        # 这里假设有一个embedding函数，实际应用中需要使用embedding模型
        query_embeddings = self._get_query_embeddings(query)

        # 调用向量数据库查询
        self.retrieved_chunks = embed_db.query(query_embeddings, self.top_k)

    def rerank(self, query: str, top_k: int = 3) -> None:
        """
        重排，赋值重排后选出的前三个最相关的片段给reranked_chunks

        Args:
            query: 用户查询
            top_k: 重排后保留的片段数量，默认为3
        """
        if not self.retrieved_chunks:
            self.reranked_chunks = []
            return

        # 这里实现重排逻辑
        # 示例：简单的基于相似度的重排
        ranked_chunks = []
        for chunk in self.retrieved_chunks:
            # 计算chunk与query的相似度（这里使用简单的关键词匹配）
            similarity = self._calculate_similarity(query, chunk)
            ranked_chunks.append((similarity, chunk))

        # 按相似度排序
        ranked_chunks.sort(reverse=True, key=lambda x: x[0])

        # 取前top_k个
        self.reranked_chunks = [chunk for _, chunk in ranked_chunks[:top_k]]

    def return_reranked_chunks(self, query: str, jcards_db: Jcards_db,
                               embed_db: Embed_db) -> List[str]:
        """
        根据用户提示词，返回最相关的三个片段

        Args:
            query: 用户查询
            jcards_db: Jcards数据库实例
            embed_db: 向量数据库实例

        Returns:
            最相关的三个片段列表
        """
        self.query = query

        # 1. 获取Jcards
        self.ask_Jcards(jcards_db)

        # 2. 筛选与查询相关的Jcards
        self.shuffle_Jcards(query)

        # 3. 召回相关片段
        self.retrieve(query, embed_db)

        # 4. 重排召回结果
        self.rerank(query)

        return self.reranked_chunks

    def _get_query_embeddings(self, query: str, RAG_write: RAG_write) -> List[float]:
        """
        将查询文本转换为向量表示

        Args:
            query: 查询文本

        Returns:
            查询向量
        """

        embeddings = RAG_write.embed_chunk(query)
        return embeddings

    def _calculate_similarity(self, query: str, chunk: str) -> float:
        """
        计算查询与片段的相似度

        Args:
            query: 查询文本
            chunk: 片段文本

        Returns:
            相似度分数
        """
        # 这里实现相似度计算逻辑
        # 示例：简单的关键词匹配相似度
        query_words = set(query.lower().split())
        chunk_words = set(chunk.lower().split())

        if not query_words:
            return 0.0

        # Jaccard相似度
        intersection = len(query_words.intersection(chunk_words))
        union = len(query_words.union(chunk_words))

        return intersection / union if union > 0 else 0.0


class Agent:
    """
    Agent类，负责处理用户请求和协调各个组件
    """

    def __init__(self):
        """初始化Agent类"""
        self.Active: bool = False  # agent是否应该主动提醒
        self.Jcards: List[str] = []  # Jcards字符串集
        self.active_content: List[str] = []  # 从主动服务类中得到的主动服务事件卡内容
        self.input: str = ""  # 用户提示词

    def process_request(self, input: str, Jcards: List[str]) -> str:
        """
         接收用户输入, 将主动警示的提示词和jcards拼在一起，调用模型api

        Args:
            input: 用户输入
            Jcards: Jcards字符串列表

        Returns:
            模型生成的回答
        """
        pass

    def active_service(self, active_service: Active_service) -> None:
        """
        对应图中"主动服务检查"，根据Active值判断是否警示，并实现警示逻辑。
        引用get_active函数向Active和active_content属性赋值

        Args:
            active_service: 主动服务实例
        """
        pass

    def Jcards_update(self, Jcards: List[str], active_content: List[str],
                      input: str) -> None:
        """
        将Jcards，提示词，主动服务卡全部喂给模型，让它判断是否需要更新Jcards。
        如果需要，则把更新内容赋值给Jcards属性，同时存储到Jcards_db。
        如果不需要则什么也不做

        Args:
            Jcards: Jcards字符串列表
            active_content: 主动服务卡内容
            input: 用户提示词
        """
        pass


class Main:
    """
    主程序类，负责程序的整体流程控制
    """

    def __init__(self):
        """初始化主程序类"""
        self.jcards_db: Jcards_db = Jcards_db()  # Jcards数据库实例
        self.embed_db: Embed_db = Embed_db()  # 向量数据库实例
        self.RAG_query: RAG_query = RAG_query()  # RAG查询实例
        self.agent: Agent = Agent()  # Agent实例

    def main_loop(self) -> None:
        """
        实现程序的主循环：输入提示词、写入、查询、输出内容、继续写入提示词，
        直到用户决定结束循环。所有属性在循环开始之前都会被定义好
        """
        pass