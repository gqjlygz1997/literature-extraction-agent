# Labeling 阶段 MVP 设计与实现

## 一句话目标

用 DSPy 生成领域配置，通过**双通道处理**（Text channel: semantic + regex → RRF；Table channel: 独立多路检索）召回候选 chunks，再用 LLM 二分类确认，为每个目标字段标注相关证据块。

当前实现支持续跑：默认跳过已经完成 labeling 的论文，`--limit N` 只处理 N 篇尚未完成的论文，
`--force` 才会重算已有 labels。

## 数据流

```
user_requirements.yaml
    ↓
DSPy 调用 LLM 生成 labeling_config.yaml
    ↓
parsed_chunks.jsonl → Chroma 向量库（增量写入）
    ↓
For each paper, for each field:
    ↓
Section Filter (exclude 硬过滤 + include 软偏好 + fallback)
    ↓
    ├─────────────────────────┬─────────────────────────┐
    ↓                         ↓                         ↓
[Text Channel]           [Table Channel]
paragraph + abstract     table chunks
    ↓                         ↓
Semantic Retrieval        如果表格 ≤5: 全部标注
    +                     如果表格 >5:
Regex Retrieval             - Table Header Keyword
    ↓                       - Table Semantic
RRF Fusion                  - Table Regex (可选)
    ↓                         ↓
Text top-5                Table RRF Fusion → Table top-5
    ↓                         ↓
    └─────────────────────────┴─────────────────────────┘
                          ↓
                DSPy 调用 LLM 二分类
                          ↓
                  labeled_chunks.jsonl
```

## 核心模块

### 1. LabelingConfigGenerator (`src/agent/labeling_config.py`)

**作用**：使用 DSPy 调用 LLM 自动生成 `labeling_config.yaml`

**输入**：`user_requirements.yaml`
- domain_description
- record.fields (name + definition)

**输出**：`labeling_config.yaml`
- 每个字段的：
  - semantic_query：自然语言检索描述句，包含 LLM 扩写出的同义词、缩写和常见表达
  - regex_patterns (pattern + description + strength)
  - table_header_keywords
  - section_include / section_exclude
  - retrieval_settings

**DSPy Signature**：
```python
class GenerateLabelingConfigSignature(dspy.Signature):
    domain_description: str
    field_name: str
    field_definition: str

    semantic_query: str
    regex_patterns_json: str  # JSON array
    table_header_keywords: str
    section_include: str
    section_exclude: str
```

### 2. VectorStoreBuilder (`src/agent/tools/vector_store.py`)

**作用**：将 `parsed_chunks.jsonl` 构建成 Chroma 向量库

**特点**：
- **增量写入**：用 chunk_id 作为 Chroma ids，避免重复 embedding
- **双文本生成**：
  - `build_table_retrieval_text()`: 轻量文本用于 embedding（caption + headers + footnotes）
  - `build_table_labeling_text()`: 完整文本用于 LLM 判断（包含 markdown_text）
- **表格字段位置**：从 `chunk["metadata"]` 读取 caption/headers/rows/markdown_text

**关键函数**：
- `build_or_update_from_chunks()`: 增量构建向量库
- `get_chunk_labeling_text(chunk_id)`: 获取完整 labeling 文本

### 3. HybridRetriever (`src/agent/tools/hybrid_retriever.py`)

#### 3.1 SectionFilter

**exclude 硬过滤**：
- references
- acknowledgements
- conflict of interest
- funding
- author contributions

**include 软偏好**（ALLMAT-style fallback）：
- 优先在 preferred chunks 中检索
- 如果候选太少，fallback 到所有 allowed chunks
- supplementary 不默认排除

**输出**：
```python
{
    "excluded": [...],     # 被 exclude 的
    "preferred": [...],    # 命中 include 的
    "fallback": [...],     # 未被 exclude 但未命中 include 的
    "allowed": [...]       # preferred + fallback
}
```

#### 3.2 SemanticRetriever

- 使用 Chroma 向量检索
- 按 paper_id 过滤（单篇论文内检索）
- 按 allowed_chunk_ids 过滤（section filter 后）
- 返回 0-based rank

#### 3.3 RegexRetriever

- 扫描所有 allowed chunks
- 按 pattern strength 打分（strong: 3.0, medium: 1.5, weak: 1.0）
- 返回 0-based rank

#### 3.4 TableHeaderRetriever

- 只检索 table chunks
- 匹配 caption/headers 中的关键词
- Caption 匹配权重高（2.0 vs 1.5）
- 返回 0-based rank

#### 3.5 RRFFusion

**公式**（0-based rank）：
```
rrf_score = sum(1 / (k + rank + 1))
```

- 默认 k = 60
- 同一个 chunk 在多个列表中出现，分数累加

### 4. EvidenceLabeler (`src/agent/tools/dspy_evidence_labeler.py`)

**作用**：DSPy 调用 LLM 对候选 chunks 做二分类

**DSPy Signature**：
```python
class EvidenceLabelSignature(dspy.Signature):
    field_name: str
    field_definition: str
    chunk_type: str
    section_path: str
    chunk_text: str  # 完整文本
    matched_patterns: str
    matched_keywords: str

    relevant: bool
```

**关键点**：
- 使用完整 chunk text（不是 preview）
- 从 `vector_store_builder.get_chunk_labeling_text()` 获取
- 只输出 `relevant`，暂不输出 confidence/reason，保持二分类简单稳定

## Text/Table 双通道

### Text Channel
- 处理：paragraph + abstract
- 流程：
  1. Section filter
  2. Semantic retrieval
  3. Regex retrieval
  4. RRF 融合
  5. Text top-5
  6. LLM 二分类

### Table Channel
- 处理：table chunks
- 流程：
  1. Section filter（exclude 硬过滤 + include 软偏好）
  2. 对所有 table chunks 计算三路检索信号（无论数量多少）：
     - TableHeaderRetriever：caption / headers keyword 匹配
     - SemanticRetriever：table semantic retrieval（allowed_chunk_ids 限定为当前论文的 table chunk ids）
     - RegexRetriever：对 table chunks 做 regex 扫描
  3. RRF 融合三路信号，生成带完整 retrieval metadata 的 ranked list
  4. 截断逻辑：
     - 如果 `table_chunks <= table_top_k`：全部 table 送 LLM，有 RRF 信号的排在前面，未被任何检索命中的追加在后（source=table_all，rrf_score=0）
     - 如果 `table_chunks > table_top_k`：只取 RRF top-k 送 LLM
  5. LLM 二分类

### 为什么分开？
- paragraph/abstract 和 table 是不同证据形态
- 表格结构在临床结局抽取中非常重要
- 不应该强行放在同一个 ranking 里

## 输出格式

### labeled_chunks.jsonl

主输出，供 Extraction 阶段使用。**一个 chunk 一行**，chunk 上挂它命中的所有 labels（对齐 ALLMAT 的思想：证据以 chunk 为单位组织，而不是 chunk-field 关系单独成行）。

只写入至少命中一个字段（`relevant=true`）的 chunk；没有任何命中的 chunk 不写入。

```json
{
  "paper_id": "PMC12345678",
  "chunk_id": "PMC12345678::a0001",
  "chunk_index": 1,
  "chunk_type": "paragraph",
  "section_path": ["Results", "Efficacy Outcomes", "Overall Survival"],
  "labels": ["treatment_regimen", "sample_size", "os"]
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `paper_id` | 论文 ID |
| `chunk_id` | chunk 唯一 ID |
| `chunk_index` | chunk 在原文中的全局序号，来自 parsed chunk 的 `metadata.chunk_index`；取不到时为 `null` |
| `chunk_type` | chunk 类型（`abstract` / `paragraph` / `table` 等） |
| `section_path` | 章节路径 |
| `labels` | 该 chunk 命中的字段列表，按 `labeling_config.yaml` 中的 field 顺序排序、去重 |

> 说明：检索与打分的细节（`rrf_score`、`semantic_similarity`、`matched_patterns`、`retrieval_channel`、`labeler_model` 等）**不再写入主输出**，以保持文件干净、便于下游消费。如需正文文本，用 `chunk_id` / `chunk_index` 回查 `parsed_chunks.jsonl`。

### labeling_summary.json

```json
{
  "total_labeled_chunks": 42,
  "total_label_assignments": 89,
  "by_field": {
    "os": {
      "chunks": 12,
      "papers": {
        "PMC12345678": 3,
        "PMC11008379": 9
      }
    }
  },
  "by_paper": {
    "PMC12345678": {
      "labeled_chunks": 6,
      "label_assignments": 14
    }
  },
  "config_fields": ["treatment_regimen", "os", "sample_size"],
  "embedding_model": "text-embedding-3-large",
  "labeler_model": "kimi-k2.6"
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `total_labeled_chunks` | 主输出的 chunk 行数（labels 非空的 chunk 数） |
| `total_label_assignments` | 所有 chunk 的 labels 总数（一个 chunk 命中 3 个字段计 3） |
| `by_field[field].chunks` | 该字段命中了多少个 chunk |
| `by_field[field].papers[paper]` | 该字段在某篇 paper 中命中了多少个 chunk |
| `by_paper[paper].labeled_chunks` | 该 paper 命中的 chunk 数 |
| `by_paper[paper].label_assignments` | 该 paper 的 label 总数 |

## 环境配置

### .env 文件

```bash
# LLM 接口配置（OpenAI 兼容协议）
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.moonshot.cn/v1
LLM_MODEL=kimi-k2.6
LLM_TEMPERATURE=1.0

# Embedding 配置（默认使用 Gemini）
GEMINI_API_KEY=your-gemini-key
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_PROVIDER=gemini

# 如需换 OpenAI embedding:
# EMBEDDING_MODEL=text-embedding-3-large
# EMBEDDING_PROVIDER=openai
# EMBEDDING_API_KEY=sk-xxx
# EMBEDDING_BASE_URL=https://api.openai.com/v1
```

## 使用示例

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 .env
cp .env.example .env
# 编辑 .env，填入 API key

# 3. 运行 labeling
python run_labeling.py \
  --requirements experiments/melanoma_trials/user_requirements.yaml \
  --chunks experiments/melanoma_trials/outputs/parsed_chunks.jsonl \
  --output experiments/melanoma_trials/outputs \
  --domain melanoma_trials \
  --limit 10
```

## 输出文件

```
experiments/melanoma_trials/outputs/
├── labeling_config.yaml          # 自动生成的配置
├── labeled_chunks.jsonl          # 主输出：一个 chunk 一行，chunk 上挂 labels
├── labeling_summary.json         # 统计摘要
└── chroma/
    └── melanoma_trials/          # Chroma 向量库
```

## 与 ALLMAT 对比

ALLMAT（Al-Alloy Materials Extraction）是本项目参考的工程先例，其 labeling 流程为：
- 使用 LangChain Embeddings + Chroma 构建向量库（默认 Gemini，也支持 OpenAI）
- 对正文段落：section 限定 → semantic top-k → (可选 regex 过滤) → LLM 二分类
- 对表格：单独处理，把表格扁平化为 text 后走 keyword/regex 判断，不混入正文 ranking

| 维度 | ALLMAT | Ours | 改进理由 |
|------|--------|------|----------|
| **配置生成** | 手写 query / regex / section 规则 | DSPy 调用 LLM 自动生成 | 减少人工成本，新领域更快上线 |
| **Text retrieval** | semantic top-k → regex 串行过滤 | semantic + regex **并行** → RRF 融合 | 串行方式 semantic 漏掉后 regex 无法补救 |
| **Table retrieval** | 表格扁平化为 text，keyword/regex 判断 | 双通道：retrieval 用轻量 text，labeling 用完整结构（含 markdown_text）；多路检索（header + semantic + regex）→ RRF | 保留表格结构供后续 extraction 使用；召回更全面 |
| **Text/Table 分离** | 两者相对分离（ALLMAT 本身有区别对待） | 显式双通道，独立 top-k，独立 labeled record | 明确隔离，`retrieval_channel` 字段可追溯 |
| **Section include** | section 白名单作为过滤条件 | 软偏好 + fallback：优先 preferred，候选不足自动扩展到 allowed | 避免 include 不准时漏掉证据 |
| **Vector store** | 使用 Chroma，重建方式不明确 | 增量写入（chunk_id 去重），再次运行不重复 embedding | 节省 embedding API 成本 |
| **LLM 调用** | 手写 prompt 字符串 | DSPy Signature + Predict | 结构化输入输出，可接 optimizer |
| **配置复用** | 不明确 | `--config` 参数 / 自动检测已有配置 | 方便手动调参后重跑实验 |

## 核心改进点

1. **Text channel**：semantic + regex 并行召回 → RRF，不是串行
2. **Table channel**：多路检索（header keyword + semantic + regex）→ RRF；表格少时全看；保留完整结构给 LLM
3. **配置自动生成**：DSPy 生成 query/regex/section，领域迁移更快
4. **增量向量库**：chunk_id 去重，避免重复 embedding
5. **完整文本标注**：LLM 拿到完整 chunk text（table 含 markdown），判断更准

## 注意事项

1. **表格字段位置**：caption/headers/rows/markdown_text 在 `chunk["metadata"]`，不是顶层字段
2. **Section include 是软偏好**：不是硬过滤，有 fallback 机制，避免过度过滤
3. **Supplementary 不默认排除**：可能包含重要的详细数据表格
4. **Chroma 增量写入**：如果 chunk 文本变了但 chunk_id 没变，需要手动清空 `chroma/` 目录重建
5. **DSPy 配置**：沿用 .env 的 LLM_* 配置，不硬编码模型名
6. **DSPy cache**：需要设置 `DSPY_CACHEDIR`（见下方环境变量说明），否则 DSPy 默认写只读路径会报错
7. **续跑语义**：同一个输出目录反复运行即可补跑新论文；修改 labeling preset 或想重算旧论文时使用 `--force`

## 下一步

1. 运行端到端测试
2. 人工抽查 labeling 质量
3. 准备进入 Extraction 阶段（使用 labeled_chunks.jsonl）
