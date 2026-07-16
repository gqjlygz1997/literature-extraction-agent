# Agent Design

## 项目目标

本项目希望构建一个可以从用户本地科学文献中自动提取结构化数据的 agent。

用户不需要为每个新领域手写一整套规则，而是提供：

- 领域说明
- 最终想要的 record 结构
- 每个 record 字段的简单定义
- 本地论文文件

系统会先根据这些输入生成论文级过滤配置；后续再逐步扩展证据定位、JSON 抽取和后处理配置。

第一阶段的 `user_requirements.yaml` 保持最小化：必填 `project_name`、`domain_description`、`record.name`、`record.meaning`、`record.fields.name` 和 `record.fields.definition`；`record.fields.type` 可选。`record.fields` 就是最终每条 record 的全部输出字段；第一版不单独写 `split_by`，任意字段值不同就保留为不同 record。用户暂时不需要手写 aliases、examples 或 glossary。

## 第一版最小闭环

当前 MVP 已经跑通从论文筛选到抽取和后处理的最小闭环：

```text
用户输入领域需求 + 本地论文
↓
Config Generator 生成 paper_filter.yaml
↓
Document Parser 轻量读取 title + abstract
↓
LLM Classifier 执行 paper filter
↓
pass / reject 结果写入 paper_filter_results.jsonl
↓
全文解析成 parsed_chunks.jsonl
↓
labeling 找到证据 chunks
↓
extraction 输出 extracted_records.jsonl
↓
postprocess 输出 records.csv
```

## 当前完整流程

```text
用户输入领域需求，提供本地论文
↓
配置生成阶段
agent 生成 paper_filter.yaml（阶段 1 只包含论文级过滤规则）
↓
论文级过滤阶段
轻量读取每篇论文的 title + abstract
↓
paper filter 判断 pass / reject
↓
reject → 跳过并记录原因
pass
↓
论文预处理阶段
完整解析全文，保留 section 层级结构
↓
清洗 + 切分段落 / 表格
↓
embedding 建向量库
↓
证据定位阶段
semantic search / regex filtering / LLM labeling
↓
结构化抽取阶段
raw extraction JSON
↓
校验与后处理阶段
clean structured table / database
```

## 批量续跑约定

推荐每个实验只用一个输出目录，例如 `experiments/pancan/outputs`。各阶段输出文件名不同，
所以可以反复复用同一个目录。

```text
--limit 10 = 本阶段最多处理 10 篇还没完成的论文
--force    = 不跳过已有结果，强制重算本阶段
```

默认情况下，已经成功完成的论文会被跳过。Extraction 阶段会自动重试失败论文，并且每完成
一篇就 checkpoint 到 `extracted_records.jsonl`，长批次中断后可以继续跑。

## 两类解析

本项目会区分两种解析：

- 轻量元数据解析：只读取 title 和 abstract，用于 paper filter。
- 完整全文解析：解析正文、章节、段落、表格和图注，用于 evidence labeling 和 extraction。

这样做的原因是：paper filter 是低成本粗筛，不应该在明显无关论文上提前做昂贵的全文切分、embedding 和 LLM 抽取。
