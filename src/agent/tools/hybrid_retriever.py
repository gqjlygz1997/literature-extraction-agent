"""
Hybrid Retriever

包含：
- SectionFilter: exclude 硬过滤 + include 软偏好（ALLMAT-style fallback）
- SemanticRetriever: 向量检索
- RegexRetriever: 正则表达式匹配检索
- TableHeaderRetriever: 表格 caption/headers 关键词匹配
- RRFFusion: 多路召回融合（0-based rank）
"""

import re
from typing import List, Dict, Set
from langchain_chroma import Chroma


class SectionFilter:
    """
    exclude: 硬过滤，明显无关章节
    include: 软偏好，优先池 + fallback
    """

    DEFAULT_EXCLUDE = [
        "references",
        "acknowledgements",
        "acknowledgments",
        "conflict of interest",
        "conflicts of interest",
        "funding",
        "author contributions"
    ]

    def __init__(self, include: List[str] = None, exclude: List[str] = None):
        self.include = [s.lower() for s in (include or [])]
        self.exclude = [s.lower() for s in (exclude or self.DEFAULT_EXCLUDE)]

    def filter_chunks(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """
        返回：
        - excluded: 被 exclude 的
        - preferred: 命中 include 的（优先池）
        - fallback: 未被 exclude 但也未命中 include 的
        - allowed: preferred + fallback（所有可用的 chunks）
        """

        excluded = []
        preferred = []
        fallback = []

        for chunk in chunks:
            section_path = chunk.get("section_path", [])
            section_text = " > ".join(section_path).lower()

            # 硬过滤：exclude
            is_excluded = False
            for excl in self.exclude:
                if excl in section_text:
                    excluded.append(chunk)
                    is_excluded = True
                    break

            if is_excluded:
                continue

            # 软偏好：include
            is_preferred = False
            if self.include:
                for incl in self.include:
                    if incl in section_text:
                        preferred.append(chunk)
                        is_preferred = True
                        break

            if not is_preferred:
                fallback.append(chunk)

        return {
            "excluded": excluded,
            "preferred": preferred,
            "fallback": fallback,
            "allowed": preferred + fallback
        }


class SemanticRetriever:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def retrieve(
        self,
        query: str,
        paper_id: str,
        allowed_chunk_ids: Set[str],
        fetch_k: int = 20
    ) -> List[Dict]:
        """
        检索单篇论文内的相关 chunks

        Args:
            query: 语义查询
            paper_id: 论文 ID
            allowed_chunk_ids: section filter 后允许的 chunk_ids
            fetch_k: 召回数量
        """

        # Chroma 按 paper_id 过滤
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=fetch_k * 2,  # 多取一些，因为后面要按 allowed_chunk_ids 过滤
            filter={"paper_id": paper_id}
        )

        ranked_results = []
        rank = 0

        for doc, distance in results:
            chunk_id = doc.metadata["chunk_id"]

            # 只保留 allowed_chunk_ids 中的
            if chunk_id not in allowed_chunk_ids:
                continue

            # distance 只作为 debug 信息
            similarity = 1 - distance  # 转换为 similarity（仅供参考）

            ranked_results.append({
                "chunk_id": chunk_id,
                "paper_id": doc.metadata["paper_id"],
                "chunk_type": doc.metadata["chunk_type"],
                "section_path_text": doc.metadata["section_path_text"],
                "semantic_rank": rank,  # 0-based
                "semantic_distance": distance,  # debug only
                "semantic_similarity": similarity,  # debug only
                "source": "semantic"
            })

            rank += 1

            if rank >= fetch_k:
                break

        return ranked_results


class RegexRetriever:
    def __init__(self, patterns: List[Dict]):
        """
        patterns: [
            {"pattern": r'\bOS\b.*?(\d+\.?\d*)\s*months?',
             "description": "OS duration",
             "strength": "strong"},
            ...
        ]
        """
        self.patterns = []
        self.failed_patterns = []

        for p in patterns:
            try:
                compiled = re.compile(p["pattern"], re.IGNORECASE)
                self.patterns.append({
                    "regex": compiled,
                    "description": p["description"],
                    "strength": p["strength"]
                })
            except re.error as e:
                # LLM 生成的 regex 可能非法，记录但不中断
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Invalid regex pattern '{p['pattern']}': {e}")
                self.failed_patterns.append({
                    "pattern": p["pattern"],
                    "description": p["description"],
                    "error": str(e)
                })

        if not self.patterns:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("No valid regex patterns compiled")

    def retrieve(
        self,
        chunks: List[Dict],
        fetch_k: int = 20
    ) -> List[Dict]:
        """扫描所有 chunks，按匹配强度排序"""

        scored_chunks = []

        for chunk in chunks:
            text = chunk.get("text", "")
            matches = []
            score = 0.0

            for pat_info in self.patterns:
                regex = pat_info["regex"]
                if regex.search(text):
                    matches.append(pat_info["description"])

                    # 强信号加分更多
                    if pat_info["strength"] == "strong":
                        score += 3.0
                    elif pat_info["strength"] == "medium":
                        score += 1.5
                    else:
                        score += 1.0

            if score > 0:
                scored_chunks.append({
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "chunk_type": chunk["chunk_type"],
                    "section_path_text": " > ".join(chunk.get("section_path", [])),
                    "regex_score": score,
                    "matched_patterns": matches,
                    "source": "regex"
                })

        # 按 score 降序排序
        scored_chunks.sort(key=lambda x: x["regex_score"], reverse=True)

        # 加上 rank（0-based）
        for rank, item in enumerate(scored_chunks[:fetch_k]):
            item["regex_rank"] = rank

        return scored_chunks[:fetch_k]


class TableHeaderRetriever:
    def __init__(self, header_keywords: List[str]):
        """
        header_keywords: 表头或 caption 中应该出现的关键词
        例如：["survival", "OS", "PFS", "response rate", "adverse events"]
        """
        self.keywords = [kw.lower() for kw in header_keywords]

    def retrieve(
        self,
        chunks: List[Dict],
        fetch_k: int = 10
    ) -> List[Dict]:
        """
        检索 table chunks，按 header/caption 匹配度排序

        注意：caption/headers 在 chunk["metadata"] 中
        """

        scored_tables = []

        for chunk in chunks:
            if chunk["chunk_type"] != "table":
                continue

            # 从 metadata 读取表格字段
            meta = chunk.get("metadata", {})
            caption = meta.get("caption", "").lower()
            headers = [h.lower() for h in meta.get("headers", [])]

            matches = []
            score = 0.0

            for kw in self.keywords:
                # Caption 匹配
                if kw in caption:
                    matches.append(f"caption:{kw}")
                    score += 2.0  # Caption 匹配权重高

                # Headers 匹配
                for header in headers:
                    if kw in header:
                        matches.append(f"header:{kw}")
                        score += 1.5
                        break  # 同一个 keyword 只加一次

            if score > 0:
                scored_tables.append({
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "chunk_type": "table",
                    "section_path_text": " > ".join(chunk.get("section_path", [])),
                    "table_score": score,
                    "matched_keywords": matches,
                    "source": "table_header"
                })

        scored_tables.sort(key=lambda x: x["table_score"], reverse=True)

        # 0-based rank
        for rank, item in enumerate(scored_tables[:fetch_k]):
            item["table_rank"] = rank

        return scored_tables[:fetch_k]


class RRFFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def merge(
        self,
        ranked_lists: List[List[Dict]],
        top_k: int = 5
    ) -> List[Dict]:
        """
        RRF 公式（0-based rank）：score = 1 / (k + rank + 1)

        ranked_lists: [
            semantic_results,    # item["semantic_rank"] = 0, 1, 2, ...
            regex_results,       # item["regex_rank"] = 0, 1, 2, ...
            table_header_results # item["table_rank"] = 0, 1, 2, ...
        ]
        """

        chunk_map = {}

        for result_list in ranked_lists:
            for item in result_list:
                chunk_id = item["chunk_id"]

                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = {
                        "chunk_id": chunk_id,
                        "paper_id": item["paper_id"],
                        "chunk_type": item["chunk_type"],
                        "section_path_text": item["section_path_text"],
                        "rrf_score": 0.0,
                        "sources": [],
                        "semantic_rank": None,
                        "semantic_similarity": None,
                        "regex_rank": None,
                        "regex_score": None,
                        "matched_patterns": [],
                        "table_rank": None,
                        "table_score": None,
                        "matched_keywords": []
                    }

                merged = chunk_map[chunk_id]
                source = item["source"]
                merged["sources"].append(source)

                # 累加 RRF score（0-based rank）
                if source == "semantic":
                    rank = item["semantic_rank"]
                    merged["rrf_score"] += 1.0 / (self.k + rank + 1)
                    merged["semantic_rank"] = rank
                    merged["semantic_similarity"] = item.get("semantic_similarity")

                elif source == "regex":
                    rank = item["regex_rank"]
                    merged["rrf_score"] += 1.0 / (self.k + rank + 1)
                    merged["regex_rank"] = rank
                    merged["regex_score"] = item.get("regex_score")
                    merged["matched_patterns"].extend(item.get("matched_patterns", []))

                elif source == "table_header":
                    rank = item["table_rank"]
                    merged["rrf_score"] += 1.0 / (self.k + rank + 1)
                    merged["table_rank"] = rank
                    merged["table_score"] = item.get("table_score")
                    merged["matched_keywords"].extend(item.get("matched_keywords", []))

        # 排序
        sorted_chunks = sorted(
            chunk_map.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        # 加上最终 hybrid rank（1-based for display）
        for i, item in enumerate(sorted_chunks[:top_k]):
            item["hybrid_rank"] = i + 1

        return sorted_chunks[:top_k]
