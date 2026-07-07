"""
DSPy Evidence Labeler

使用 DSPy 调用 LLM 对候选 chunks 做二分类确认
判断 chunk 是否包含目标字段的相关证据
"""

import dspy
import os
from dotenv import load_dotenv
from typing import Dict, List
import logging


logger = logging.getLogger(__name__)


class EvidenceLabelSignature(dspy.Signature):
    """Classify whether a chunk is relevant evidence for a target field."""

    field_name: str = dspy.InputField(
        desc="Name of the target extraction field"
    )
    field_definition: str = dspy.InputField(
        desc="Definition of what information this field should contain"
    )
    chunk_type: str = dspy.InputField(
        desc="Type of chunk: paragraph, table, abstract"
    )
    section_path: str = dspy.InputField(
        desc="Section path where this chunk appears"
    )
    chunk_text: str = dspy.InputField(
        desc="The FULL text content of the chunk to evaluate"
    )
    matched_patterns: str = dspy.InputField(
        desc="Regex patterns that matched this chunk, if any"
    )
    matched_keywords: str = dspy.InputField(
        desc="Table header keywords that matched, if any"
    )

    relevant: bool = dspy.OutputField(
        desc="True if this chunk contains relevant evidence for the field"
    )


class EvidenceLabeler:
    def __init__(self, vector_store_builder):
        """
        Args:
            vector_store_builder: VectorStoreBuilder 实例，用于获取完整 chunk text
        """
        load_dotenv()

        # 从 .env 读取 LLM 配置
        model_name = os.getenv("LLM_MODEL", "gpt-4o")
        api_key = os.getenv("LLM_API_KEY")
        api_base = os.getenv("LLM_BASE_URL")
        # Kimi k2.6 在 JSON mode 下要求 temperature=1.0
        temperature = 1.0 if "kimi" in model_name.lower() else float(os.getenv("LLM_TEMPERATURE", "0.0"))

        # Kimi 可能把 reasoning_content 单独返回；二分类也需要留足 JSON wrapper 空间。
        max_tokens = int(os.getenv("LABELER_MAX_TOKENS", "1500"))
        self.lm = dspy.LM(
            model=f"openai/{model_name}",
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens
        )
        dspy.settings.configure(lm=self.lm)
        self.predictor = dspy.Predict(EvidenceLabelSignature)

        # 用于获取完整 chunk text
        self.vector_store_builder = vector_store_builder

    def label_chunk(
        self,
        field_name: str,
        field_definition: str,
        candidate: Dict
    ) -> Dict:
        """对单个候选 chunk 做二分类"""

        # 获取完整 labeling text（不是 preview）
        chunk_id = candidate["chunk_id"]
        full_text = self.vector_store_builder.get_chunk_labeling_text(chunk_id)

        # 如果获取不到完整文本，尝试从 candidate 中获取
        if not full_text:
            full_text = candidate.get("text", "")

        try:
            result = self.predictor(
                field_name=field_name,
                field_definition=field_definition,
                chunk_type=candidate["chunk_type"],
                section_path=candidate["section_path_text"],
                chunk_text=full_text,  # 完整文本
                matched_patterns=", ".join(candidate.get("matched_patterns", [])),
                matched_keywords=", ".join(candidate.get("matched_keywords", []))
            )
            return {"relevant": bool(result.relevant)}
        except Exception as exc:
            logger.warning(
                "Evidence labeling failed for chunk=%s field=%s; keeping candidate relevant. Error: %s",
                chunk_id,
                field_name,
                exc,
            )
            return {"relevant": True}

    def label_candidates(
        self,
        field_config: Dict,
        candidates: List[Dict],
        channel: str
    ) -> List[Dict]:
        """
        批量标注候选 chunks

        Args:
            field_config: 字段配置
            candidates: 候选 chunks
            channel: "text" 或 "table"
        """

        labeled = []
        for candidate in candidates:
            label_result = self.label_chunk(
                field_name=field_config["field_name"],
                field_definition=field_config["field_definition"],
                candidate=candidate
            )

            # 合并结果，添加 channel 信息
            labeled_chunk = {
                **candidate,
                **label_result,
                "retrieval_channel": channel
            }
            labeled.append(labeled_chunk)

        return labeled
