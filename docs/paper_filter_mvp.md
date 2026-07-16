# Paper-Filter MVP 设计文档

> 本文档记录第一阶段 MVP 的最终设计决策和实现约定。
> 代码实现位于 `src/agent/`，CLI 入口为 `run_paper_filter.py`。

---

## 数据流

```
experiments/<domain>/user_requirements.yaml   ← 用户手写
        │
        ▼  ConfigGenerator.generate()          (LLM，一次性)
outputs/paper_filter.yaml                     ← 落盘，可用 --config 复用
        │
        ▼  DocumentParser.parse_metadata_light (每篇，只读 title + text_for_filter)
ArticleMeta{title, abstract, text_for_filter, metadata_quality, ...}
        │
        ▼  LLMLabeler.classify_paper           (每篇，按 criteria 逐条判三态)
PaperFilterDecision{decision, criteria, reason, model}
        │
        ▼  Workflow._write_outputs
outputs/paper_filter_results.jsonl
outputs/passed_papers.jsonl
outputs/rejected_papers.jsonl
outputs/run_summary.json
```

阶段输出支持续跑：默认复用已有 `paper_filter_results.jsonl`，只处理还没有结果的论文。
`--limit N` 限制的是本次最多新增处理 N 篇，不是固定取输入文件前 N 篇。`--force` 会重算
已有论文。

---

## 用户输入格式：`user_requirements.yaml`

用户手写领域需求和最终 record 结构，不要求手写 `aliases`、`example`、`glossary` 或 `output_preferences`。

必填：
- `project_name`
- `domain_description`
- `record.name`
- `record.meaning`
- `record.fields[].name`
- `record.fields[].definition`

可选：
- `record.fields[].type`，不写时默认为 `string`

`record.fields` 是最终每条结构化 record 里要输出的全部字段。第一版不再单独写 `split_by`：如果任意 record field 的值不同，后续 extraction / post-processing 默认保留为不同 record。缩写、同义说法和单位说明直接写在 `definition` 里。

```yaml
project_name: pancan_treatment_outcomes

domain_description: >
  Extract treatment regimens and survival or efficacy outcomes from local
  full-text pancreatic cancer clinical treatment papers.

record:
  name: treatment_outcome_record
  meaning: >
    One record represents one patient group or treatment arm within one paper.
    It includes the patient group, sample size, disease stage, treatment regimen,
    treatment setting, and reported outcomes for that same group.

  fields:
    - name: patient_group
      definition: Patient group or trial arm.
      type: string

    - name: treatment_regimen
      definition: Treatment or intervention received by this patient group.
      type: string

    - name: os
      definition: OS = overall survival for this patient group and treatment regimen.
      type: number
```

完整示例见：
- [`experiments/pancan/user_requirements.yaml`](../experiments/pancan/user_requirements.yaml)
- [`experiments/hea/user_requirements.yaml`](../experiments/hea/user_requirements.yaml)

---

## 生成配置：`paper_filter.yaml`

**阶段 1 只生成到 `paper_filter` 为止**，`retrieval` / `extraction` 不生成。

```yaml
domain_name: pancan_treatment_outcomes
domain_description: ...
target_fields:
  - {name: os, description: "OS = overall survival，总生存期，优先记录 median OS，单位通常为 months。", final_type: number}
field_definitions:
  os: OS = overall survival，总生存期，优先记录 median OS，单位通常为 months。
  pfs: PFS = progression-free survival，无进展生存期，优先记录 median PFS，单位通常为 months。
paper_filter:
  input_scope: title_and_abstract
  inclusive_when_uncertain: true
  pass_condition: all_required_not_false
  criteria:
    - name: is_pancreatic_cancer
      question: Is this paper about pancreatic cancer (in humans)?
      rationale: 领域归属；非胰腺癌直接排除。
      required: true
    - name: is_clinical_treatment_study
      question: Is this a primary clinical treatment study in human patients?
      rationale: 排除综述、机制、动物实验。
      required: true
    - name: may_report_treatment_outcomes
      question: >
        Might this paper report treatment regimens together with survival or
        efficacy outcomes such as OS, PFS, ORR, DCR, or HR?
      rationale: 粗判是否可能含目标结局，用 "might" 不判断具体数值。
      required: true
```

---

## 文件解析策略

### 支持格式

| 格式 | 处理方式 |
|------|---------|
| XML (JATS/PMC) | 结构化提取：`front/abstract` 所有 `<p>` 拼合 |
| HTML | 启发式提取（见下方优先级） |
| PDF | **不支持**，记录 `error: PDF is not supported`，继续处理其他文件 |

### `ArticleMeta.metadata_quality` 四级

| 值 | 含义 | LLM 处理 |
|----|------|---------|
| `structured_xml` | JATS XML，title + abstract 完整 | 正常分类 |
| `html_abstract` | HTML，成功提取 abstract | 正常分类 |
| `html_front_matter` | HTML，abstract 缺失，用前 5 段代替 | uncertain → pass |
| `title_only` | 只有 title | uncertain → pass |
| `parse_error` | 解析失败或 PDF | 直接记录 error，不分类 |

### HTML 提取优先级

**title**（按顺序尝试）：
1. `meta[name=citation_title]`
2. `meta[name=dc.title]`
3. `meta[property=og:title]`
4. 带 `article-title` class/id 的 heading 元素
5. `<title>` 标签，去掉 `| Journal` / `- Publisher` 后缀

**abstract**（按顺序尝试）：
1. `meta[name=citation_abstract]`
2. `meta[name=dc.description]`（长度 > 100 字符）
3. 带 `abstract` 在 id/class 里的 `section`/`div`
4. 标题为 "Abstract" / "Summary" 的 `h2`/`h3`/`h4` 后方段落

**front_matter fallback**（abstract 缺失时）：
- 去除 nav/header/footer/script/style 后，取前 5 个长度 > 80 字符的 `<p>`
- 合并，截取最多 2000 字符
- `front_matter_used: true`，`metadata_quality: html_front_matter`

---

## 三态 Criteria 设计

每条 criterion 的 LLM 输出是一个对象：

```json
"is_pancreatic_cancer": {
  "answer": "true | false | uncertain",
  "reason": "一句话理由"
}
```

**Decision 规则**（`pass_condition: all_required_not_false`）：

| 条件 | 结果 |
|------|------|
| 任何 `required` criterion 答案为 `"false"` | **reject** |
| 所有 `required` criteria 答案为 `"true"` 或 `"uncertain"` | **pass** |
| LLM 响应无法解析 | **pass**，`reason: parse_failed_pass_by_default` |
| title 和 text_for_filter 均为空 | **pass**，`reason: empty_metadata_pass_by_default` |

> **设计原则**：paper filter 是早期粗筛，目标高召回。
> `uncertain` 显式保留在输出中，用于人工审计"勉强放过"的论文。
> 后续统计时 uncertain 按 pass 处理，但输出中不压缩为 bool。

---

## 输出文件格式

### `paper_filter_results.jsonl`（每行一篇）

```json
{
  "source_path": "experiments/pancan/input_papers/PMC7257856.xml",
  "file_type": "xml",
  "paper_id": "PMC7257856",
  "doi": "10.xxxx/xxxx",
  "title": "Nab-paclitaxel plus gemcitabine ...",
  "abstract_available": true,
  "front_matter_used": false,
  "metadata_quality": "structured_xml",
  "decision": "pass",
  "criteria": {
    "is_pancreatic_cancer": {"answer": "true",      "reason": "..."},
    "is_clinical_treatment_study": {"answer": "uncertain", "reason": "..."},
    "may_report_treatment_outcomes": {"answer": "true", "reason": "..."}
  },
  "reason": "No required criterion is explicitly false.",
  "model": "kimi-k2.6"
}
```

错误行（parse_error）：

```json
{
  "source_path": "...",
  "file_type": "unknown",
  "paper_id": "paper",
  "doi": "", "title": "",
  "abstract_available": false,
  "front_matter_used": false,
  "metadata_quality": "parse_error",
  "decision": "error",
  "criteria": {}, "reason": "", "model": "",
  "error": "PDF is not supported"
}
```

### `passed_papers.jsonl`（精简，供下游定位文件）

```json
{"paper_id": "PMC7257856", "source_path": "...", "file_type": "xml",
 "doi": "...", "title": "...", "abstract_available": true, "metadata_quality": "structured_xml"}
```

### `rejected_papers.jsonl`（含 criteria 方便人工复核）

```json
{"paper_id": "...", "source_path": "...", "title": "...",
 "criteria": {...}, "reason": "Rejected because: is_pancreatic_cancer: ..."}
```

### `run_summary.json`

```json
{"domain_name": "...", "total": 42, "passed": 30, "rejected": 9, "error": 3, "dry_run": 0,
 "config_path": ".../paper_filter.yaml"}
```

---

## CLI 参数

```bash
# 主方式：requirements 文件
python run_paper_filter.py \
  --requirements experiments/pancan/user_requirements.yaml \
  --input        experiments/pancan/input_papers \
  --output       experiments/pancan/outputs \
  --limit        10

# Fallback：命令行直传
python run_paper_filter.py \
  --domain  "从本地胰腺癌临床治疗论文抽取..." \
  --fields  "treatment_regimen,os,pfs" \
  --field-definitions "os=overall survival;pfs=progression-free survival" \
  --input   experiments/pancan/input_papers \
  --output  experiments/pancan/outputs \
  --limit   10

# 复用已有 config（跳过 LLM 生成）
python run_paper_filter.py \
  --config  experiments/pancan/outputs/paper_filter.yaml \
  --input   experiments/pancan/input_papers \
  --output  experiments/pancan/outputs \
  --limit   10

# 冒烟：只跑前 3 篇，不调用 LLM
python run_paper_filter.py \
  --requirements experiments/pancan/user_requirements.yaml \
  --input   experiments/pancan/input_papers \
  --output  experiments/pancan/outputs \
  --dry-run --limit 3
```

| 参数 | 说明 |
|------|------|
| `--requirements PATH` | user_requirements.yaml 路径（推荐） |
| `--config PATH` | 已生成的 paper_filter.yaml，跳过 LLM 生成 |
| `--domain TEXT` | 领域说明（fallback） |
| `--fields f1,f2,...` | 字段列表（fallback） |
| `--field-definitions k=v;k=v` | 字段定义（fallback） |
| `--input DIR` | 论文目录（必填） |
| `--output DIR` | 输出目录（必填） |
| `--model NAME` | 覆盖 LLM 模型 |
| `--limit N` | 最多处理 N 篇尚未完成的论文 |
| `--force` | 不跳过已有结果，重算本阶段 |
| `--dry-run` | 只解析 metadata，不调用 LLM |
| `--verbose` | DEBUG 日志 |

> PDF 无配置选项。遇到 PDF 直接记录 `error: PDF is not supported` 并继续处理其他文件。
> `--dry-run` 不生成新配置、不调用 LLM，主要用于快速检查本地 XML/HTML 的 title/abstract 能否被读出。

## 续跑行为

推荐固定使用同一个输出目录，例如 `experiments/pancan/outputs`。再次运行时：

- 已经在 `paper_filter_results.jsonl` 里的论文会被跳过。
- `passed_papers.jsonl` 和 `rejected_papers.jsonl` 会由旧结果 + 本次新结果重新生成。
- `--limit 10` 会补跑 10 篇还没有 paper-filter 结果的论文。
- 如果修改了 `paper_filter.yaml` 或想重新判断旧论文，加 `--force`。

---

## 模块职责边界

| 模块 | 职责 | 不负责 |
|------|------|--------|
| `document_parser.py` | 轻量提取 title/abstract/front_matter；判断 metadata_quality | 全文解析、embedding |
| `config_generator.py` | LLM 生成 paper_filter criteria；save/load config yaml | retrieval/extraction schema |
| `llm_labeler.py` | 按 criteria 分类论文；三态结果；决策逻辑 | 证据定位、段落标注 |
| `workflow.py` | 编排：扫描 → 解析 → 分类 → 写输出 | 无 |
| `user_requirements.py` | 加载/校验 user_requirements.yaml；CLI fallback 组装 | 无 |

---

## 与 ALLMAT/Sisyphus 的关系

ALLMAT 的 `paper_filter` 是三分类（`hea_experimental / hea_theoretical / irrelevant`）加一个 `mechanical_relevancy` bool，没有 `uncertain` 态。

本系统面向用户本地论文，metadata 质量更不稳定，因此：
- 显式保留 `uncertain` 三态，有利于高召回和人工审计
- 增加 `front_matter_used` / `metadata_quality` 字段，追踪输入质量
- `inclusive_when_uncertain: true` 是硬约束，写进 config，不可配置

HTML 提取策略借鉴了 `JatsLoader.peek_meta` 的结构理念，但实现独立在 agent 内，不 import pancan 项目。

---

## 已接入的后续阶段

- `run_fulltext_acquisition.py`：WOS 路线下 DOI/PMID → PMCID → PMC XML
- `run_preprocess.py`：完整 section 层级解析，输出 `parsed_chunks.jsonl`
- `run_labeling.py`：语义/正则/表格检索，输出 `labeled_chunks.jsonl`
- `run_extraction.py`：结构化 JSON 抽取，输出 `extracted_records.jsonl`
- `run_postprocess.py`：规范化、过滤、去重和 CSV 导出
