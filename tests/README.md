# 三层次用户记忆测试

## 为何测试被 SKIP？

若看到 **4 skipped** 且原因为 `sentence_transformers required for RAG/embedding`，说明当前 Python 环境未安装 RAG/嵌入依赖，`embed_db` 会自动跳过，导致所有依赖向量库的用例被跳过。

## 如何让测试真正执行

1. **使用已装好依赖的环境**（推荐）  
   在已安装 `sentence-transformers` 等依赖的 conda/venv 中运行，例如：
   ```bash
   conda activate Better_User_Memory_2026_2   # 或你的项目环境名
   python -m pytest tests/ -v
   ```

2. **在当前环境安装依赖**  
   在项目根目录执行：
   ```bash
   pip install -r requirements.txt
   ```
   若遇权限问题，可加 `--user` 或使用虚拟环境：
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python -m pytest tests/ -v
   ```

安装完成后，再运行 `python -m pytest tests/ -v`，四个用例应会执行而非跳过。
