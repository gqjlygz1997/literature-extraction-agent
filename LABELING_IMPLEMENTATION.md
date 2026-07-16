# Labeling 阶段实现完成

## 已实现的模块

### 1. 核心模块

#### `src/agent/labeling_config.py`
- `LabelingConfigGenerator`: 使用 DSPy 调用 LLM 自动生成配置
- `GenerateLabelingConfigSignature`: DSPy Signature 定义
- 从 `user_requirements.yaml` 生成 `labeling_config.yaml`

#### `src/agent/tools/vector_store.py`
- `VectorStoreBuilder`: 增量构建 Chroma 向量库
- `build_chunk_labeling_text()`: 生成完整 labeling 文本
- `build_table_labeling_text()`: 表格完整文本（含 markdown）
- `build_table_retrieval_text()`: 表格轻量文本（用于 embedding）
- `chunk_to_langchain_doc()`: DocumentChunk → LangChain Document

#### `src/agent/tools/hybrid_retriever.py`
- `SectionFilter`: exclude 硬过滤 + include 软偏好（ALLMAT-style fallback）
- `SemanticRetriever`: 向量检索（按 paper_id 和 allowed_chunk_ids 过滤）
- `RegexRetriever`: 正则表达式匹配检索
- `TableHeaderRetriever`: 表格 caption/headers 关键词匹配
- `RRFFusion`: 多路召回 RRF 融合（0-based rank: 1/(k+rank+1)）

#### `src/agent/tools/dspy_evidence_labeler.py`
- `EvidenceLabeler`: DSPy 调用 LLM 做二分类
- `EvidenceLabelSignature`: DSPy Signature 定义
- 使用完整 chunk text（不是 preview）

### 2. Workflow 集成

#### `src/agent/workflow.py`
- `run_labeling_mvp()`: Labeling 阶段主流程
- `_process_paper_field_labeling()`: 双通道处理（text + table）
- `_generate_labeling_summary()`: 生成统计摘要
- 支持 resume：默认跳过已完成论文，`--limit N` 只处理尚未完成的 N 篇，`--force` 重算

### 3. 入口脚本

#### `run_labeling.py`
- CLI 入口脚本
- 参数验证和错误处理
- 日志输出
- `--limit` / `--force` 批处理参数

### 4. 文档和测试

#### `docs/labeling_mvp.md`
- 完整设计文档
- 数据流说明
- 与 ALLMAT 对比
- 使用示例

#### `test_labeling_basic.py`
- 基本功能测试
- SectionFilter 测试
- RRFFusion 测试
- 表格 metadata 读取测试

## 关键设计决策

### 1. 按审查要求修改的地方

✅ **表格字段读取位置**：从 `chunk["metadata"]` 读取 caption/headers/rows/markdown_text

✅ **Section include/exclude 逻辑**：
- exclude 是硬过滤（明显无关章节）
- include 是软偏好（ALLMAT-style fallback）
- supplementary 不默认排除

✅ **Chroma 增量写入**：
- 用 chunk_id 作为 Chroma ids
- 检查已存在的 chunk_id，跳过重复 embedding
- `build_or_update_from_chunks()` 实现增量逻辑

✅ **Text/Table 双通道**：
- Text channel: paragraph + abstract → semantic + regex → RRF → text top-5
- Table channel: table chunks → table header → table top-5
- 两个通道独立处理，结果合并
- 每条记录保留 `retrieval_channel` 字段

✅ **DSPy/LLM 配置**：
- 沿用现有 .env 的 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL/LLM_TEMPERATURE
- 不硬编码模型名
- Embedding 默认使用 Gemini，可选切换 OpenAI

✅ **RRF 公式**：
- 统一为 0-based rank: `1 / (k + rank + 1)`
- 默认 k=60

### 2. 核心改进点

1. **并行 hybrid retrieval**：semantic + regex + table header 三路并行召回 → RRF 融合
2. **双通道处理**：text 和 table 不混在同一个 ranking 里
3. **配置自动生成**：DSPy 生成 query/regex/section，降低人工成本
4. **增量向量库**：避免重复 embedding，节省成本
5. **结构化表格**：保留表格结构，后续 extraction 可用

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件：

```bash
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.moonshot.cn/v1
LLM_MODEL=kimi-k2.6
LLM_TEMPERATURE=1.0

# Embedding: Google Gemini
GEMINI_API_KEY=your-gemini-key
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_PROVIDER=gemini

# 可选：OpenAI Embedding
# EMBEDDING_MODEL=text-embedding-3-large
# EMBEDDING_PROVIDER=openai
# EMBEDDING_API_KEY=sk-xxx
# EMBEDDING_BASE_URL=https://api.openai.com/v1
```

### 3. 运行 Labeling

```bash
python run_labeling.py \
  --requirements experiments/melanoma_trials/user_requirements.yaml \
  --chunks experiments/melanoma_trials/outputs/parsed_chunks.jsonl \
  --output experiments/melanoma_trials/outputs \
  --domain melanoma_trials \
  --limit 10
```

### 4. 输出文件

```
experiments/melanoma_trials/outputs/
├── labeling_config.yaml          # 自动生成的配置
├── labeled_chunks.jsonl          # 标注结果（主输出）
├── labeling_summary.json         # 统计摘要
└── chroma/
    └── melanoma_trials/          # Chroma 向量库（可重用）
```

## 测试

### 基本功能测试

```bash
python test_labeling_basic.py
```

测试内容：
- 所有模块导入
- SectionFilter 逻辑（exclude + include fallback）
- RRFFusion 逻辑（0-based rank）
- 表格 metadata 读取

## 注意事项

1. **表格字段位置**：caption/headers/rows/markdown_text 在 `chunk["metadata"]`，不是顶层
2. **Section include 是软偏好**：不是硬过滤，有 fallback 机制，避免过度过滤
3. **Supplementary 不排除**：可能包含重要的详细数据表格
4. **Chroma 增量写入**：如果 chunk 文本变了但 chunk_id 没变，需要手动清空 `chroma/` 目录重建
5. **Semantic retrieval 过滤**：检索后按 `allowed_chunk_ids` 过滤，确保 section filter 生效
6. **续跑语义**：同一个输出目录可反复补跑；如果改了 labeling 配置或想重算旧论文，加 `--force`

## 下一步

1. **端到端测试**：用真实数据运行完整流程
2. **人工抽查**：验证 labeling 质量
3. **Extraction 阶段**：已实现，可直接读取 `labeled_chunks.jsonl` 和 `parsed_chunks.jsonl`

## 文件清单

### 新增文件
- `src/agent/labeling_config.py`
- `src/agent/tools/vector_store.py`
- `src/agent/tools/hybrid_retriever.py`
- `src/agent/tools/dspy_evidence_labeler.py`
- `run_labeling.py`
- `docs/labeling_mvp.md`
- `test_labeling_basic.py`

### 修改文件
- `src/agent/workflow.py`：添加 `run_labeling_mvp()` 和相关辅助函数
- `requirements.txt`：添加 jsonlines, langchain-core, langchain-openai, langchain-chroma, chromadb
- `.env`：添加 Embedding 配置注释

## 与 ALLMAT 对比总结

| 维度 | ALLMAT | Ours | 优势 |
|------|--------|------|------|
| 配置生成 | 手写 | DSPy 自动生成 | 领域迁移更快 |
| Retrieval | 串行（semantic→regex） | 并行（semantic+regex→RRF） | 避免漏掉 |
| Table 处理 | 扁平化 | 保留结构 | 后续可用 |
| Text/Table | 混合 | 双通道 | 证据形态不同 |
| Section include | 硬过滤 | 软偏好+fallback | 避免过滤 |
| Vector store | 不明确 | 增量写入 | 节省成本 |
| LLM 调用 | 手写 prompt | DSPy Signature | 结构化 I/O |

## 实现完成 ✅

所有核心模块已实现，包括：
- DSPy 配置生成
- 增量向量库构建
- 三路并行检索 + RRF 融合
- 双通道处理（text + table）
- LLM 二分类确认
- 完整的 workflow 集成

可以开始端到端测试和验证！
