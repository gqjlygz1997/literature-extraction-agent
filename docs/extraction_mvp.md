# Extraction Stage MVP

## 概述

Stage 2 Extraction 从 Labeling 阶段的输出（`labeled_chunks.jsonl`）出发，结合原始文本（`parsed_chunks.jsonl`），通过 LLM 的结构化输出接口，一次性抽取 `user_requirements.yaml` 中定义的所有字段，输出 `extracted_records.jsonl`。

本阶段对应 ALLMAT 的 **Contextualized Extraction** 策略（`sisyphus/heas/extract_lc.py`），对 ALLMAT 工程方法做减法：

- ✅ 保留：Contextualized extraction、动态 JSON Schema、三段式 Prompt、`with_structured_output(method="json_schema")`、Record cleanup
- ⚠️ 暂不实现：DetectProcesses / 模板注入（`synthesis.py`），保留扩展位置
- ❌ 不做：paragraph-wise extraction、多策略对比、DSPy optimizer、Stage 3 Post-processing

---

## 快速开始

```bash
python run_extraction.py \
  --requirements experiments/pancan/user_requirements.yaml \
  --chunks       experiments/pancan/outputs/parsed_chunks.jsonl \
  --labels       experiments/pancan/outputs/labeled_chunks.jsonl \
  --output       experiments/pancan/outputs \
  --limit        10
```

输出目录：

```
experiments/pancan/outputs/
├── extracted_records.jsonl   # 一行一条 record（已去重）
└── extraction_summary.json   # 统计摘要
```

`--limit 10` 表示本次最多抽取 10 篇还没完成的论文；已经成功抽取过的论文会跳过。
失败论文不会被当作完成，下一次会自动重试。只有加 `--force` 才会重算已有成功结果。

---

## 数据流

```
parsed_chunks.jsonl + labeled_chunks.jsonl + user_requirements.yaml
        ↓
  [context_builder]
  收集 labeled chunks + abstract chunks
  按 chunk_index 排序（原文顺序）
  渲染为带 header 的文本块
        ↓
  [extraction_schema]
  根据 record.fields 动态生成 Pydantic Records 模型
  根据 record.name / record.meaning / fields 生成三段式 Prompt
        ↓
  [extractor]
  LLM.with_structured_output(Records, method="json_schema")
  一次性抽取所有 records
        ↓
  [record_cleanup]
  同 paper 内严格规则去重
  字段全量补齐（缺失字段 → null）
  分配 record_id，附加每条 record 最相关的 source_chunk_ids
        ↓
  extracted_records.jsonl
```

---

## 输出格式

### extracted_records.jsonl

每行一条记录，字段顺序固定：`paper_id` → `record_id` → 所有 `record.fields`（按 yaml 顺序）→ `source_chunk_ids`。

```json
{
  "paper_id": "PMC10389558",
  "record_id": "PMC10389558::r0001",
  "patient_group": "neoadjuvant therapy group",
  "sample_size": "2959",
  "disease_stage": "Stage I-III PDAC",
  "treatment_regimen": "neoadjuvant therapy followed by surgery",
  "line_of_therapy": "neoadjuvant",
  "os": "24.3 months",
  "pfs": null,
  "orr": null,
  "dcr": null,
  "hr": "0.82",
  "ci": "0.71-0.94",
  "p_value": "<0.001",
  "source_chunk_ids": ["PMC10389558::a0001", "PMC10389558::p0009"]
}
```

**字段说明：**

| 字段 | 说明 |
|---|---|
| `paper_id` | 来源论文 ID |
| `record_id` | `{paper_id}::r{n:04d}`，同 paper 内从 r0001 开始编号 |
| `record.fields` | 来自 `user_requirements.yaml`，缺失字段为 `null`，所有字段必须出现 |
| `source_chunk_ids` | 每条 record 最可能来自的 chunk ids；由 value / endpoint / treatment / statistics 等字段反查上下文得到，通常比整篇 extraction 上下文更短 |

> **重要**：所有字段值均为字符串（`Optional[str]`），即使 yaml 中定义为 `type: number`。数值规范化在 Stage 3 Post-processing 处理。

### extraction_summary.json

```json
{
  "total_papers_processed": 2,
  "total_papers_failed": 0,
  "total_records_extracted": 7,
  "duplicates_removed": 1,
  "processed_this_run": 2,
  "skipped_existing": 8,
  "by_paper": {
    "PMC10389558": {
      "extraction_status": "ok:plain_json",
      "records_raw": 4,
      "records_after_cleanup": 3,
      "duplicates_removed": 1,
      "context_chunks_used": 6
    }
  },
  "extractor_model": "kimi-k2.6",
  "timestamp": "2026-07-07T10:00:00+00:00"
}
```

---

## Prompt 设计

三段式结构，**全部动态生成，不含领域硬编码词汇**，跨领域可用（pancan / HEA 等）。

### System Message

```
You are a precise scientific data extraction assistant.

Your task is to extract structured records from scientific paper excerpts.

A {record.name} is defined as:
{record.meaning}

Rules:
- Create one record per distinct {record.name}.
- If any field value differs between two records, keep them as separate records.
- Only extract information explicitly stated in the text.
- ...
```

`{record.name}` 和 `{record.meaning}` 直接来自 `user_requirements.yaml`。

### User Message

```
[START OF EVIDENCE CONTEXT]
{context}
[END OF EVIDENCE CONTEXT]

Instruction:
{instruction}
```

### Instruction（动态生成）

```
Extract all {record.name} records from the evidence context above.

Each record must contain the following fields:

  - **patient_group**: Patient group or trial arm...
  - **os**: Overall survival...
  ...

Additional rules:
  - If a field is not explicitly stated, output null.
  - Create separate records for different values.
  ...
```

> **扩展位置**：`extraction_schema.create_instruction()` 末尾预留了 DetectProcesses 模板注入位置（注释标记），与 ALLMAT `synthesis.py` 对应。

---

## Context Building

对应 ALLMAT：`reorder_paras()` + `ParagraphExtend.from_paragraphs()`

```python
# 收集策略
labeled_chunk_ids  →  来自 labeled_chunks.jsonl（命中至少一个 field 的 chunk）
abstract_chunk_ids →  来自 parsed_chunks.jsonl（chunk_type == "abstract"）

# 排序：按 chunk_index 升序（原文顺序）
# 截断策略：abstract 优先不被截断；labeled 按顺序截断

# 渲染格式
[CHUNK {chunk_id} | {chunk_type} | {section_path}]
{body}
```

`body` 渲染规则：
- `abstract` / `paragraph`：直接用 `text`
- `table`：`Caption: {caption}\n{markdown_text}`

---

## Record Cleanup

对应 ALLMAT：`entity_resolution_rule()` + `partition_strict()` 的简化版

- 只在同一 `paper_id` 内处理
- normalize：strip + 折叠空格 + 统一小写
- 所有字段 normalize 后完全相同 → duplicate，只保留一条
- 任意字段不同 → 保留为独立 record
- **不做**：fuzzy merge、LLM entity resolution、跨 paper 合并、数值换算、领域标准化

空值规则：
- `null` 和 `""` normalize 后均为 `""`，视为相同
- `null` vs 非空值 → **不合并**（信息量不同）

---

## JSON Schema / Pydantic 动态生成

对应 ALLMAT：`create_result_model_dynamic()`

```python
Record = create_model("Record", __base__=BaseModel, **field_defs)
Record.model_config = ConfigDict(extra="forbid")   # → additionalProperties: false

Records = create_model("Records", __base__=BaseModel,
    records=(List[Record], Field(...)))
Records.model_config = ConfigDict(extra="forbid")
```

`model_config` 通过属性赋值而不是传给 `create_model` 的 kwargs，避免 Pydantic v2 中被当成普通字段。

**类型策略**：第一版全部 `Optional[str]`（含 yaml 中 `type: number` 的字段）。数值转换在 Stage 3 处理。

---

## 与 ALLMAT 的对照

| 我们的实现 | 对应 ALLMAT |
|---|---|
| `extraction_schema.build_records_model()` | `extract_lc.create_result_model_dynamic()` |
| `extraction_schema.create_instruction()` | `prompt.create_instruction_dynamic()` |
| `extraction_schema.build_system_message()` | `prompt.SYSTEM_MESSAGE_NO_SYN` |
| `tools/context_builder.build_context()` | `reorder_paras()` + `ParagraphExtend.from_paragraphs()` |
| `tools/extractor.extract_records()` | `template | model.with_structured_output(method="json_schema")` |
| `tools/record_cleanup.deduplicate()` | `entity_resolution_rule()` + `partition_strict()` |
| 暂不实现 | `synthesis.get_synthesis_prompt()` / `DetectProcesses` |

---

## 环境变量

```bash
# LLM 配置（支持 OpenAI 兼容协议，如 Kimi）
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.moonshot.cn/v1
LLM_MODEL=kimi-k2.6

# Extraction 可单独覆盖
EXTRACTOR_MODEL=kimi-k2.6
EXTRACTOR_MAX_TOKENS=4000
EXTRACTOR_TIMEOUT=90
EXTRACTOR_MAX_RETRIES=0

# Labeling 阶段沿用
EMBEDDING_MODEL=gemini-embedding-001
```

`EXTRACTOR_MAX_TOKENS` 太低时，模型可能写到一半被截断，状态会变成
`failed:LengthFinishReasonError`。这种论文不会被标记为完成；调高 token 上限后重新运行
同一条命令即可补跑。

## 续跑 / checkpoint

Extraction 每完成一篇都会把当前累计结果写回 `extracted_records.jsonl`，最后再写
`extraction_summary.json`。续跑时：

- `ok:*` 和 `skipped:no_context` 视为已完成，会跳过。
- `failed:*` 不视为完成，会自动重试。
- 本次待处理论文会先从旧 `extracted_records.jsonl` 中移除，再写入新结果，避免同一论文重复记录。
- `--force` 会关闭跳过逻辑，用当前输入和 prompt 重算。

---

## 测试

```bash
# 单元测试（不调 LLM）
python test_extraction_basic.py

# 端到端（调真实 LLM）
python run_extraction.py \
  --requirements experiments/pancan/user_requirements.yaml \
  --chunks       experiments/pancan/outputs/parsed_chunks.jsonl \
  --labels       experiments/pancan/outputs/labeled_chunks.jsonl \
  --output       experiments/pancan/outputs \
  --limit        10
```
