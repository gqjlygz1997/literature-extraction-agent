"""
Vector Store Builder

将 parsed_chunks.jsonl 构建成 Chroma 向量库
- 增量写入，避免重复 embedding
- 使用 chunk_id 作为 Chroma ids
- 保留原始 chunk 映射，用于获取完整 labeling text
"""

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import Dict, List
import jsonlines
import os
from pathlib import Path
from dotenv import load_dotenv


def build_chunk_labeling_text(chunk: Dict) -> str:
    """
    为 chunk 生成用于 labeling 的完整文本
    这个文本会传给 EvidenceLabeler，不是 preview
    """
    if chunk["chunk_type"] == "table":
        return build_table_labeling_text(chunk)
    else:
        return chunk.get("text", "")


def build_table_labeling_text(table_chunk: Dict) -> str:
    """
    Table 的完整 labeling text
    从 chunk["metadata"] 读取表格字段
    包含 caption + headers + markdown + footnotes
    """
    meta = table_chunk.get("metadata", {})

    parts = []

    caption = meta.get("caption", "")
    if caption:
        parts.append(f"Table Caption: {caption}")

    headers = meta.get("headers", [])
    if headers:
        parts.append(f"Column Headers: {' | '.join(headers)}")

    markdown_text = meta.get("markdown_text", "")
    if markdown_text:
        parts.append(f"\nTable Content:\n{markdown_text}")

    footnotes = meta.get("footnotes", [])
    if footnotes:
        parts.append(f"\nTable Notes: {' '.join(footnotes)}")

    return "\n".join(parts) if parts else ""


def build_table_retrieval_text(table_chunk: Dict) -> str:
    """
    Table 的轻量 retrieval text（用于 embedding）
    从 chunk["metadata"] 读取表格字段
    只包含 caption + headers + footnotes
    不包含完整 markdown，避免稀释信号
    """
    meta = table_chunk.get("metadata", {})

    parts = []

    caption = meta.get("caption", "")
    if caption:
        parts.append(f"Table: {caption}")

    headers = meta.get("headers", [])
    if headers:
        parts.append(f"Columns: {', '.join(headers)}")

    footnotes = meta.get("footnotes", [])
    if footnotes:
        parts.append(f"Notes: {' '.join(footnotes)}")

    return "\n".join(parts) if parts else ""


def chunk_to_langchain_doc(chunk: Dict) -> Document:
    """DocumentChunk → LangChain Document"""

    # 根据 chunk_type 选择 retrieval text
    if chunk["chunk_type"] == "table":
        page_content = build_table_retrieval_text(chunk)
    else:
        page_content = chunk.get("text", "")

    # Chroma metadata：只放扁平字段
    section_path = chunk.get("section_path", [])
    metadata = {
        "paper_id": chunk["paper_id"],
        "chunk_id": chunk["chunk_id"],
        "chunk_type": chunk["chunk_type"],
        "section_path_text": " > ".join(section_path),
        "section_0": section_path[0] if len(section_path) > 0 else "",
        "section_1": section_path[1] if len(section_path) > 1 else "",
    }

    return Document(page_content=page_content, metadata=metadata)


class VectorStoreBuilder:
    def __init__(self, embedding_config: Dict, persist_dir: str):
        load_dotenv()

        model = embedding_config["model"]
        provider = (embedding_config.get("provider") or "").lower()

        if provider == "gemini" or model.startswith("gemini-"):
            from .gemini_embeddings import GeminiEmbeddings
            self.embeddings = GeminiEmbeddings(model=model)
        else:
            # 优先使用 EMBEDDING_* 配置，fallback 到 LLM_* 配置
            api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")
            api_base = os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL")

            self.embeddings = OpenAIEmbeddings(
                model=model,
                api_key=api_key,
                base_url=api_base
            )
        self.persist_dir = persist_dir

        # 保存原始 chunk 映射（chunk_id -> full chunk）
        # 后续 labeling 需要用完整 text
        self.chunk_map = {}

    def build_or_update_from_chunks(self, chunks_path: str) -> Chroma:
        """
        增量构建或更新向量库，避免重复 embedding

        逻辑：
        - 用 chunk_id 作为 Chroma ids
        - 写入前检查 chunk_id 是否已存在
        - 已存在则跳过，不重新 embedding
        - 不存在则 add_documents
        """

        # 加载所有 chunks
        print(f"Loading chunks from: {chunks_path}")
        all_chunks = []
        with jsonlines.open(chunks_path) as reader:
            for chunk in reader:
                self.chunk_map[chunk["chunk_id"]] = chunk
                all_chunks.append(chunk)

        print(f"Loaded {len(all_chunks)} chunks")

        # 检查 Chroma 是否已存在
        persist_path = Path(self.persist_dir)

        if persist_path.exists() and any(persist_path.iterdir()):
            print(f"Loading existing vector store from {self.persist_dir}")
            vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )

            # 获取已有的 chunk_ids
            existing_ids = set()
            try:
                collection = vectorstore._collection
                all_docs = collection.get()
                if all_docs and "ids" in all_docs:
                    existing_ids = set(all_docs["ids"])
            except Exception as e:
                print(f"Warning: Failed to get existing IDs: {e}")
                existing_ids = set()

            print(f"Found {len(existing_ids)} existing chunks in vector store")

            # 找出需要新增的 chunks
            new_chunks = [
                chunk for chunk in all_chunks
                if chunk["chunk_id"] not in existing_ids
            ]

            if new_chunks:
                print(f"Adding {len(new_chunks)} new chunks...")
                new_docs = [chunk_to_langchain_doc(c) for c in new_chunks]
                new_ids = [c["chunk_id"] for c in new_chunks]

                vectorstore.add_documents(documents=new_docs, ids=new_ids)
                print(f"✓ Added {len(new_chunks)} new chunks")
            else:
                print("✓ No new chunks to add, vector store is up to date")

        else:
            print(f"Creating new vector store at {self.persist_dir}")
            persist_path.mkdir(parents=True, exist_ok=True)

            documents = [chunk_to_langchain_doc(c) for c in all_chunks]
            ids = [c["chunk_id"] for c in all_chunks]

            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                ids=ids
            )
            print(f"✓ Created vector store with {len(documents)} chunks")

        return vectorstore

    def get_chunk_labeling_text(self, chunk_id: str) -> str:
        """获取用于 labeling 的完整文本"""
        chunk = self.chunk_map.get(chunk_id)
        if not chunk:
            return ""
        return build_chunk_labeling_text(chunk)

    def get_chunk(self, chunk_id: str) -> Dict:
        """获取原始 chunk"""
        return self.chunk_map.get(chunk_id, {})
