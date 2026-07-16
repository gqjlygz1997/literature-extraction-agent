# Labeling 阶段快速开始指南

## 前置条件

1. 已完成 Paper Filter 阶段，得到 `passed_papers.jsonl`
2. 已完成 Preprocessing 阶段，得到 `parsed_chunks.jsonl`
3. 有 `user_requirements.yaml` 文件

## 一键运行

```bash
python run_labeling.py \
  --requirements experiments/YOUR_DOMAIN/user_requirements.yaml \
  --chunks experiments/YOUR_DOMAIN/outputs/parsed_chunks.jsonl \
  --output experiments/YOUR_DOMAIN/outputs \
  --domain YOUR_DOMAIN \
  --limit 10
```

## 示例

假设你的领域是 `melanoma_trials`：

```bash
python run_labeling.py \
  --requirements experiments/melanoma_trials/user_requirements.yaml \
  --chunks experiments/melanoma_trials/outputs/parsed_chunks.jsonl \
  --output experiments/melanoma_trials/outputs \
  --domain melanoma_trials \
  --limit 10
```

## 输出文件

运行完成后，会在输出目录生成：

```
experiments/melanoma_trials/outputs/
├── labeling_config.yaml          # LLM 自动生成的配置
├── labeled_chunks.jsonl          # 主输出：一个 chunk 一行，chunk 上挂 labels
├── labeling_summary.json         # 统计摘要
└── chroma/
    └── melanoma_trials/          # 向量库（可重用）
```

## 流程说明

### Step 1: 生成配置
LLM 读取 `user_requirements.yaml` 里的 `record.fields`，自动生成每个字段的：
- semantic_query：自然语言检索描述句，包含同义词、缩写和常见表达
- regex_patterns
- table_header_keywords
- section_include/exclude

### Step 2: 构建向量库
将 `parsed_chunks.jsonl` 构建成 Chroma 向量库：
- **增量写入**：第二次运行时，只添加新 chunks
- **节省成本**：避免重复 embedding

### Step 3-5: 双通道标注
对每篇论文、每个字段：

**Text Channel** (paragraph + abstract):
```
Section Filter → Semantic + Regex → RRF → Text top-5 → LLM 二分类
```

**Table Channel** (table chunks):
```
Section Filter → Table Header Match → Table top-5 → LLM 二分类
```

### Step 6-7: 输出结果
- `labeled_chunks.jsonl`: 主输出，**一个 chunk 一行**，chunk 上挂它命中的所有字段（`labels`）。只写入 labels 非空的 chunk。每行字段：`paper_id` / `chunk_id` / `chunk_index` / `chunk_type` / `section_path` / `labels`。
- `labeling_summary.json`: 统计 `total_labeled_chunks` / `total_label_assignments`、每个字段命中的 chunk 数（`by_field`）、每篇 paper 命中的 chunk 与 label 数（`by_paper`）。

示例（`labeled_chunks.jsonl` 一行）：
```json
{"paper_id": "PMC10389558", "chunk_id": "PMC10389558::a0001", "chunk_index": 1, "chunk_type": "abstract", "section_path": ["Abstract", "Results:"], "labels": ["treatment_regimen", "sample_size", "os"]}
```

## 预期耗时

假设：
- 5 篇论文
- 每篇 50 chunks
- 3 个字段

**时间分解**：
1. 配置生成: ~30 秒（3 个字段 × LLM 调用）
2. 向量库构建: ~2 分钟（250 chunks × embedding）
3. 检索 + 标注: ~5 分钟（5 papers × 3 fields × (retrieval + 5 LLM calls)）

**总计**: ~7-8 分钟

**再次运行**: 默认跳过已完成论文，只处理新论文；向量库已存在时会继续增量写入。

## 常见问题

### Q1: 如何查看 labeling 结果？

```bash
# 查看摘要
cat experiments/melanoma_trials/outputs/labeling_summary.json | python -m json.tool

# 查看前 5 条标注记录（一个 chunk 一行）
head -5 experiments/melanoma_trials/outputs/labeled_chunks.jsonl | python -m json.tool

# 查看命中某个字段的 chunks
grep '"overall_survival"' labeled_chunks.jsonl
```

### Q2: 向量库占用多少空间？

Chroma 向量库大小取决于 chunks 数量和 embedding 维度：
- 100 chunks × 3072 维: ~1.2 MB
- 1000 chunks × 3072 维: ~12 MB

### Q3: 如何清空向量库重建？

```bash
rm -rf experiments/melanoma_trials/outputs/chroma/
```

然后重新运行 `run_labeling.py`。

### Q4: 如何修改配置？

**方式 1**: 手动编辑生成的 `labeling_config.yaml`，然后重新运行

**方式 2**: 修改 `user_requirements.yaml` 中的 `record.fields` 字段定义，删除 `labeling_config.yaml`，重新运行

### Q5: 如何调整 top-k 数量？

编辑 `labeling_config.yaml` 中的 `retrieval_settings`:

```yaml
retrieval_settings:
  semantic_fetch_k: 20    # semantic 召回数量
  regex_fetch_k: 20       # regex 召回数量
  table_top_k: 5          # table 最终数量
  text_top_k: 5           # text 最终数量
  rrf_k: 60               # RRF 参数
```

### Q6: 如何查看哪些 chunks 被标注为相关？

```python
import json

with open('experiments/melanoma_trials/outputs/labeled_chunks.jsonl') as f:
    for line in f:
        record = json.loads(line)
        # 主输出只包含 labels 非空的 chunk，一个 chunk 一行
        print(f"Paper: {record['paper_id']}")
        print(f"Chunk: {record['chunk_id']} (index={record['chunk_index']})")
        print(f"Section: {' > '.join(record['section_path'])}")
        print(f"Labels: {record['labels']}")
        print("-" * 60)
```

> 主输出不含正文文本。需要 chunk 正文时，用 `chunk_id` / `chunk_index` 回查 `parsed_chunks.jsonl`。

## 下一步

完成 Labeling 后，进入 **Extraction 阶段**：

```bash
python run_extraction.py \
  --requirements experiments/melanoma_trials/user_requirements.yaml \
  --chunks experiments/melanoma_trials/outputs/parsed_chunks.jsonl \
  --labels experiments/melanoma_trials/outputs/labeled_chunks.jsonl \
  --output experiments/melanoma_trials/outputs \
  --limit 10
```

Extraction 已实现，并且和 labeling 一样支持续跑。修改 labeling 配置或想重算旧论文时，
对 labeling 阶段使用 `--force`。

## 故障排除

### 错误: "No module named 'dspy'"

```bash
pip install dspy-ai
```

### 错误: "No module named 'langchain_chroma'"

```bash
pip install langchain-chroma chromadb
```

### 错误: "API key not found"

检查 `.env` 文件是否正确设置：

```bash
cat .env | grep LLM_API_KEY
```

### 错误: "Chroma collection already exists"

这不是错误，是正常的增量写入提示。如果想强制重建：

```bash
rm -rf experiments/melanoma_trials/outputs/chroma/
```

## 测试

运行基本功能测试：

```bash
python test_labeling_basic.py
```

应该看到：
```
✅ All basic tests passed!
```

## 需要帮助？

查看完整文档：
- `docs/labeling_mvp.md`: 完整设计文档
- `IMPLEMENTATION_SUMMARY.md`: 实现总结
