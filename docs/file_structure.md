# File Structure

这份文档解释当前项目骨架里每个目录和文件的全称与用途。

## 顶层目录

| 名称 | 全称 | 中文意思 | 用途 |
| --- | --- | --- | --- |
| `docs/` | documents / documentation | 项目文档 | 放设计说明、流程图说明、实验设计、给导师看的文字材料 |
| `src/` | source / source code | 源代码 | 放真正实现 agent 和工具模块的代码 |
| `configs/` | configurations | 配置文件 | 放领域抽取配置示例，例如胰腺癌和高熵合金 |
| `experiments/` | experiments | 实验目录 | 放实验输入论文、人工标注、运行结果和评估报告 |
| `examples/` | examples | 示例需求文件 | 放可以直接复制改写的 `user_requirements.yaml` |
| `presets/` | presets | 领域预设 | 放稳定的 paper filter、labeling、extraction、postprocess 配置 |

## 文档文件

| 文件 | 全称 | 用途 |
| --- | --- | --- |
| `README.md` | read me | 项目入口说明，告诉自己和别人这个项目是做什么的 |
| `requirements.txt` | requirements text file | 记录运行当前项目需要安装的 Python 包 |
| `docs/agent_design.md` | agent design document | 记录 agent 的目标、输入输出、阶段划分和当前 MVP 流程 |
| `docs/file_structure.md` | file structure document | 解释每个目录和文件的全称、中文意思和用途 |
| `docs/quickstart.md` | quick start document | 从安装到完整跑通的阶段命令 |
| `docs/pipeline_overview.md` | pipeline overview document | 每个阶段的数据流、输出文件和续跑语义 |
| `docs/*_mvp.md` | MVP design documents | 各阶段设计与实现细节 |

## 核心代码

| 文件 | 全称 | 用途 |
| --- | --- | --- |
| `src/agent/__init__.py` | package initializer | 让 `agent` 成为一个 Python 包，后续可以被 import |
| `src/agent/config_schema.py` | configuration schema | 定义第一阶段 `paper_filter.yaml` 应该包含哪些字段和结构 |
| `src/agent/config_generator.py` | configuration generator | 根据用户领域需求、目标字段和示例，调用 LLM 生成第一阶段 `paper_filter.yaml` |
| `src/agent/workflow.py` | workflow controller | 总流程调度器，负责按顺序调用各个工具模块 |
| `run_*.py` | run scripts | 阶段级命令行入口，例如 paper filter、preprocess、labeling、extraction |

## 工具模块

| 文件 | 全称 | 用途 |
| --- | --- | --- |
| `src/agent/tools/document_parser.py` | document parser | 解析本地 XML / HTML，先轻量读 title 和 abstract，后续再完整解析全文；不支持 PDF，遇到 PDF 直接记录 error |
| `src/agent/tools/pmc_downloader.py` | PMC downloader | 从 DOI/PMID 查 PMCID 并下载 PMC JATS/XML |
| `src/agent/tools/retriever.py` | evidence retriever | 根据 semantic query 从段落或表格中找候选证据 |
| `src/agent/tools/regex_filter.py` | regular expression filter | 根据配置中的关键词或正则表达式过滤候选段落 |
| `src/agent/tools/llm_labeler.py` | large language model labeler | 调用 LLM 判断论文、段落或表格是否满足某个条件 |
| `src/agent/tools/extractor.py` | structured extractor | 根据 JSON schema 和抽取 prompt，把证据内容转成原始结构化 JSON |
| `src/agent/tools/verifier.py` | result verifier | 检查抽取结果是否符合 schema，数值是否有证据支持 |

## 配置示例

| 文件 | 全称 | 用途 |
| --- | --- | --- |
| `configs/pancan.example.yaml` | pancreatic cancer example configuration | 胰腺癌任务的 `paper_filter.yaml` 示例 |
| `configs/hea.example.yaml` | high-entropy alloy example configuration | 高熵合金任务的 `paper_filter.yaml` 示例 |

## 实验目录

| 目录或文件 | 全称 | 用途 |
| --- | --- | --- |
| `experiments/pancan/` | pancreatic cancer experiment | 胰腺癌方向实验材料和结果 |
| `experiments/hea/` | high-entropy alloy experiment | 高熵合金方向实验材料和结果 |
| `input_papers/` | input papers | 用户提供的本地论文文件放这里 |
| `outputs/` | outputs | 系统生成的结果文件放这里 |

推荐一个实验只用一个 `outputs/` 目录。各阶段会写不同文件名，`--limit 10` 会补跑本阶段
还没完成的 10 篇论文，已经完成的论文默认跳过；只有 `--force` 才重算旧结果。
