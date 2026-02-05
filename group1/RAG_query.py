from typing import List, Tuple, Optional, Any
import json
import sys
from pathlib import Path

# 添加 group3 到路径以导入 embed_chunk
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from group3.rag_ingest_incremental import embed_chunk as group3_embed_chunk


class Jcards_db:
    """Jcards数据库接口类"""

    def get_Jcards_tostr(self) -> List[str]:
        """
        从jcards库中获取所有jcards，返回字符串列表形式
        """
        pass

class RAG_write:
    def embed_chunk(query: str) -> List[float]:
        """将query转换为向量表示"""
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


    def retrieve(self, query: str, embed_db: Embed_db, top_k: int = 5) -> None:
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


    def rerank(self, query: str, retrieved_chunks: List[str], top_k: int = 3) -> None:
        """
        重排，赋值重排后选出的前三个最相关的片段给reranked_chunks

        Args:
            query: 用户查询
            top_k: 重排后保留的片段数量，默认为3
        """
        if not self.retrieved_chunks:
            self.reranked_chunks = []
            print("召回结果为空！请先进行召回！")
            return
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        pairs = [(query, chunk) for chunk in retrieved_chunks]
        scores = cross_encoder.predict(pairs)

        scored_chunks = list(zip(retrieved_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        self.reranked_chunks = [chunk for chunk, _ in scored_chunks][:top_k]



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

        # 3. 召回相关片段
        self.retrieve(self.query, embed_db, 5)

        # 4. 重排召回结果
        self.rerank(self.query, self.retrieved_chunks, 3)

        return self.reranked_chunks

    def _get_query_embeddings(self, query: str) -> List[float]:
        """
        将查询文本转换为向量表示（使用 group3 的 embed_chunk）

        Args:
            query: 查询文本

        Returns:
            查询向量
        """
        return group3_embed_chunk(query)


