import os
import queue
import threading
import time
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量
try:
    load_dotenv()
except FileNotFoundError:
    print("警告：未找到 .env 文件，将使用系统环境变量。")
except Exception as e:
    print(f"警告：加载 .env 文件时出错: {e}")

class LLMCompatibleClient:
    """
    用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        self.stream_read_timeout = int(os.getenv("LLM_STREAM_READ_TIMEOUT", 120))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    # route: 1-1-2 调用大语言模型进行思考，并返回其响应。 messages: 提示词集  temperature： 温度
    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        para:
        messages: List[Dict[str, str]]提示词集, 格式如下：

        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            # 流式读取带超时：子线程往队列放 chunk，主线程带超时取，避免“响应成功”后长时间无内容
            chunk_queue = queue.Queue()
            stream_error = []

            def consume_stream():
                try:
                    for chunk in response:
                        content = chunk.choices[0].delta.content or ""
                        if content:
                            chunk_queue.put(content)
                    chunk_queue.put(None)
                except Exception as e:
                    stream_error.append(e)
                    chunk_queue.put(("__error__", e))

            reader = threading.Thread(target=consume_stream, daemon=True)
            reader.start()
            collected_content = []
            start_time = time.monotonic()
            first_chunk = True
            while True:
                remaining = self.stream_read_timeout - (time.monotonic() - start_time)
                if remaining <= 0:
                    print("\n❌ 流式读取超时：超过 {} 秒未完成。".format(self.stream_read_timeout))
                    return None
                try:
                    item = chunk_queue.get(timeout=min(60, remaining))
                except queue.Empty:
                    print("\n❌ 流式读取超时：等待下一块内容超时。")
                    return None
                if item is None:
                    break
                if isinstance(item, tuple) and item[0] == "__error__":
                    print(f"\n❌ 调用LLM API时发生错误: {item[1]}")
                    return None
                if first_chunk:
                    print("✅ 大语言模型响应成功:")
                    first_chunk = False
                print(item, end="", flush=True)
                collected_content.append(item)
            print()
            if stream_error:
                print(f"❌ 流式读取过程中发生错误: {stream_error[0]}")
                return None
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None


# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = LLMCompatibleClient(
            model="deepseek-chat",
            apiKey="sk-55950ea43bc44fb58e5379fc9f2c1d2a",
            baseUrl="https://api.deepseek.com",
            timeout=60
        )

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
