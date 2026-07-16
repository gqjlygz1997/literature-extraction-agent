# Preprocessing MVP 设计文档

> 第二阶段：把通过 paper filter 的 PMC/JATS XML 全文，解析成保留 section 层级的
> 统一内容块（DocumentChunk），做轻量清洗和段落/表格切分，落盘成 JSONL，供后续 labeling 使用。
>
> 本阶段**不做** embedding、regex filtering、LLM labeling、extraction。
> 代码实现位于 `src/agent/`，CLI 入口为 `run_preprocess.py`。

---

## 0. 定位与边界

| 必须做 | 暂时不做 |
|--------|---------|
| XML/JATS 全文解析 | HTML / PDF 全文 |
| section_path 层级保留 | row_texts（行级文本块） |
| paragraph / abstract / table chunk | 行级表格检索 |
| ALLMAT-style 段落切分 | embedding |
| 表格保留 section_path + caption + label + header_rows + headers + rows + markdown_text | LLM labeling |
| 输出 parsed_chunks.jsonl + preprocessing_summary.json | extraction |

**分层原则（沿用 ALLMAT）**：预处理只做结构性清洗，**领域相关性判断全部留给 labeling**。
参考 `sisyphus/heas/label.py:13` 的 `label_properties_restricted`——introduction/conflict/support
的过滤是 label 阶段的领域策略，不是预处理。预处理保留这些 section，但把 `sec_type`
写进 metadata，让 labeling 自己决定滤不滤，避免误删 evidence。
**特别地，supplementary-material 不默认删**（常含数据表格，是核心 evidence 来源），
只删纯结构性无 evidence 的 ref-list / ack / fn-group / notes。见 §5。

---

## 1. 数据流

```
experiments/<domain>/outputs/passed_papers.jsonl   ← 上一阶段（paper filter）产物
        │
        ▼  DocumentParser.parse_full_text(path)     (每篇，全文解析)
list[DocumentChunk]  (abstract / paragraph / table)
        │
        ▼  Workflow.run_preprocess_mvp
experiments/<domain>/outputs/parsed_chunks.jsonl        ← 所有 chunk，一行一个
experiments/<domain>/outputs/preprocessing_summary.json ← 统计
```

只处理 `passed_papers.jsonl` 里 `file_type == "xml"` 的论文；HTML/PDF 记 error 跳过。

---

## 2. 统一内容块 DocumentChunk

沿用 `document_parser.py` 已有定义，只放宽 metadata 类型：

```python
@dataclass(frozen=True)
class DocumentChunk:
    paper_id: str
    chunk_id: str
    text: str
    chunk_type: Literal["paragraph", "table", "abstract", "caption", "title"]
    section_path: list[str]        # 栈式 list，比 ALLMAT 的 "/" 字符串更干净
    metadata: dict                 # dict[str, Any]（表格要塞 list/int）
```

**唯一结构改动**：`metadata: dict[str, str]` → `dict[str, Any]`（表格 headers/rows/n_rows 需要非字符串值）。

### 与 ALLMAT 的对应关系

ALLMAT 用 LangChain `Document`（仅 `page_content` + `metadata`），把类型塞进
`metadata['sub_titles']`（段落存 section 路径、摘要存 `"Abstract"`、表格存 `"table"`），
是"一字段三用"。我们的改进：

| 我们的字段 | ALLMAT 对应 | 改进 |
|-----------|------------|------|
| `paper_id` | `metadata['source']` / `doi` | 显式主键 |
| `text` | `page_content` | 一致 |
| `chunk_type` | 隐含在 `sub_titles` | **独立字段**，不复用 |
| `section_path` | `sub_titles` 字符串（`/` 拼） | **list**，任意嵌套深度 |
| `metadata` | `source/doi/title` | 表格额外保留结构化字段 |

### 字段填充约定

| 字段 | paragraph | table | abstract |
|------|-----------|-------|----------|
| `chunk_type` | `"paragraph"` | `"table"` | `"abstract"` |
| `text` | 清洗后段落文本 | `"{label}. {caption}\n{markdown_text}"` | 摘要文本 |
| `section_path` | 如 `["Results", "Bulk-tensile"]` | 表格所在 section（**改进点**） | `["Abstract"]` 或 `["Abstract","RESULTS"]` |

### metadata 公共字段（所有 chunk）

```
doi, title, source_path, sec_type, chunk_index, token_count
```
- `sec_type`：JATS `<sec sec-type="...">` 原生属性（`introduction/results/conclusion/...`），可能为 `null`
- `chunk_index`：本篇内全局出现序号（从 0 递增）

### 表格 chunk 额外 metadata（比 ALLMAT 多存）

```
label        : "Table 1"
caption      : "Average phase-specific hardness and modulus values..."
header_rows  : [["Phase","Hardness","Modulus"]]   # <thead> 原始多行，list[list[str]]
headers      : ["Phase","Hardness","Modulus"]      # header_rows 拍平成单层列名
rows         : [["FCC","3.2","180"], ...]          # <tbody> 数据行，list[list[str]]
n_rows, n_cols
footnotes    : [...]                                # <table-wrap-foot>，可为空
markdown_text: "| Phase | ... |\n|---|...|"         # 也是 text 的表体部分
```

**为什么多存**：
- `caption` 单独存 → labeling 可只读 caption 判断，不必读整表
- `header_rows` 保留原始多行表头（JATS 常见两层表头），`headers` 提供拍平版方便直接用
- `rows` 结构化保留 → 为第二版行级抽取留接口，本版不用但零成本
- `markdown_text` → 保证表格能进 embedding / 喂 LLM

---

## 3. XML 全文解析逻辑（JATS）

深度优先遍历，栈式维护 section_path。对齐 ALLMAT `chempp` 的
`ArticleElementType`（SECTION_TITLE / PARAGRAPH / TABLE）思路，但**只写 JATS 一种**，
不照搬 `chempp` 的多出版社 HTML parser。

```
parse_full_text(path):
    soup = BeautifulSoup(xml, "lxml-xml")
    paper_id, doi, title = 复用 parse_metadata_light 的 ID 提取
    body = soup.find("body")
    if body is None: raise ParseError("no <body> found")
    chunks = []
    emit_abstract(front, chunks)          # front/abstract
    walk(body, section_path=[], chunks)   # body 递归
    return chunks

walk(sec, section_path, chunks):
    title_el = sec.find("title", recursive=False)
    cur_path = section_path + [title_el.text] if title_el else section_path
    if should_skip_section(sec, cur_path):   # §5 清洗
        record_skip(sec); return
    for child in sec.children (recursive=False, 仅 Tag):
        if child.name == "sec":         walk(child, cur_path, chunks)
        elif child.name == "p":         emit_paragraph(child, cur_path, chunks)
        elif child.name == "table-wrap": emit_table(child, cur_path, chunks)
        # fig / disp-formula 等第一版跳过
```

**要点：**
- **section_path 用栈式 list**：比 ALLMAT `title_hierarchy` 数组 + `/` join 干净，
  天然支持任意嵌套深度（ALLMAT 卡在 h1–h6 六层）。
- **`<p>` 内嵌 `<table-wrap>`**：JATS 常见（ALLMAT `xml_section_extract_acs:87` 先 `pop` 出来）。
  emit_paragraph 时先把内嵌 table-wrap 抽出单独 emit，段落文本用剩余部分。
- **摘要**：普通 abstract 一段一块 `["Abstract"]`；结构化 abstract（BACKGROUND/METHODS/
  RESULTS/…）把子标题接进 section_path，如 `["Abstract","RESULTS"]`。

---

## 4. 表格转文本（改进 ALLMAT）

主 `text` 用 **Markdown 表格**（LLM 理解优于 CSV，caption 单独成行），
同时保留结构化字段。

```
emit_table(table_wrap, section_path, chunks):
    label   = table_wrap.find("label")            # "Table 1"
    caption = table_wrap.find("caption") 文本
    grid, header_rows, rows = parse_jats_table(table_wrap.find("table"))
    headers = flatten(header_rows)                # 拍平单层列名
    markdown = to_markdown(headers, rows)
    text = f"{label}. {caption}\n{markdown}"       # 进 embedding / 喂 LLM
    footnotes = table_wrap.find_all(...table-wrap-foot...)
    metadata = {label, caption, header_rows, headers, rows,
                n_rows, n_cols, footnotes, markdown_text=markdown, ...公共字段}
    chunks.append(DocumentChunk(chunk_type="table", text=text,
                                section_path=section_path, metadata=metadata))
```

`parse_jats_table` **复用 ALLMAT `loader.py:38` `parse_html_table_to_json` 的
rowspan/colspan 网格填充算法**，只把输出从 CSV 改成 `(header_rows, rows)` 二维结构：
- 建 `len(rows) × maxcols` 网格
- 遍历 `<tr>` 的 `<th>/<td>`，按 `rowspan`/`colspan` 填充，合并单元格用 `""` 占位
- `<thead>` 行归入 `header_rows`，`<tbody>` 行归入 `rows`（无 thead 时首行作 header）

**关键改进 vs ALLMAT**：`section_path` 真实保留表格所在章节（ALLMAT 直接丢成 `"table"`）。
labeling 阶段既能语义检索表格，也能按 section 过滤。

---

## 5. 清洗策略

**原则**：只删"结构性无关"内容；领域相关性判断全部留给 labeling。

**预处理跳过（保守、纯结构性，明确无 evidence 的）：**
- `<back>` 里的 `ref-list`（参考文献）、`ack`（致谢）、`fn-group`、`notes`
- 空段落、纯符号/纯 citation 残留段落

**预处理保留（打 `sec_type` 标，交给 labeling 决定）：**
- Introduction、Methods/Experimental、Conflict of interest、Funding、Author contributions
- **`sec-type == "supplementary-material"` 的 section —— 必须保留**

> ⚠️ **supplementary 不默认删**：补充材料常含数据表格（成分表、力学性能表等），
> 是后续论文实验的核心 evidence 来源。默认删除风险过高。
> supplementary 里的 section/table 照常解析成 chunk，`sec_type` 标为
> `supplementary-material`，是否使用交给 labeling 阶段决定。

**开关（第一版写死，不进 config）：**
- `drop_reference_and_ack = True`（只删 ref-list / ack / fn-group / notes）
- `keep_all_body_sections = True`（body 内不按语义删，含 supplementary）

---

## 6. 段落切分策略（ALLMAT-style）

对齐 `loader.py:157` 的 `chunk_text`：

- 段落 token ≤ 400 → 不切，整段一个 chunk
- 段落 > 400 token → 按句子累积切分，每块 ~400 token，避免尾块 < 200 token（防碎块）
- 目标区间 **200–600 token**

**依赖决策（与 ALLMAT 的差异）：**

| 项 | ALLMAT | 本项目 | 理由 |
|----|--------|--------|------|
| token 计数 | `tiktoken` cl100k_base | **引入 tiktoken** | 轻量准确，下游迟早要用 |
| 句子切分 | `nltk.sent_tokenize`（需下载 punkt） | **正则切句** `(?<=[.!?])\s+(?=[A-Z])` | 免 nltk 数据下载；科技文献缩写多，正则够用 |

→ `requirements.txt` 新增 `tiktoken`。

**chunk_id 稳定性：** `f"{paper_id}::{prefix}{chunk_index:04d}"`
- 段落 `p`、表格 `t`、摘要 `a`；如 `PMC7021732::p0012`、`PMC7021732::t0001`
- 全局序号按解析出现顺序递增，保证重复解析结果稳定可复现

---

## 7. 输出文件

单一 chunk 流 + summary，**不拆 parsed_tables.jsonl**（`chunk_type` 已能区分，
下游统一遍历最简单）。

```
experiments/<domain>/outputs/
  ├── parsed_chunks.jsonl        # 所有 passed 论文的所有 chunk，一行一个
  └── preprocessing_summary.json # 统计
```

默认续跑：如果某篇论文已经在 `parsed_chunks.jsonl` 中出现，本阶段会跳过它。
`--limit N` 只计算尚未解析的论文；`--force` 会重算本阶段并替换对应论文的 chunks。

### parsed_chunks.jsonl — 段落行

```json
{"paper_id":"PMC7021732","chunk_id":"PMC7021732::p0012","chunk_type":"paragraph",
 "text":"The tensile strength increased from 320 MPa to 480 MPa after aging at 800 °C for 2 h...",
 "section_path":["Results and Discussion","Bulk-tensile and compression"],
 "metadata":{"doi":"10.1039/xxxx","title":"...",
             "source_path":"experiments/hea/input_papers/PMC7021732.xml",
             "sec_type":"results","chunk_index":12,"token_count":118}}
```

### parsed_chunks.jsonl — 表格行

```json
{"paper_id":"PMC7021732","chunk_id":"PMC7021732::t0001","chunk_type":"table",
 "text":"Table 1. Average phase-specific hardness and modulus values from nano-indentation.\n| Phase | Hardness (GPa) | Modulus (GPa) |\n|---|---|---|\n| FCC | 3.2 | 180 |\n| BCC | 5.1 | 210 |",
 "section_path":["Results and Discussion","Small-scale mechanical behavior by nano-indentation"],
 "metadata":{"doi":"10.1039/xxxx","title":"...","source_path":"...",
             "sec_type":null,"chunk_index":27,"token_count":64,
             "label":"Table 1",
             "caption":"Average phase-specific hardness and modulus values from nano-indentation.",
             "header_rows":[["Phase","Hardness (GPa)","Modulus (GPa)"]],
             "headers":["Phase","Hardness (GPa)","Modulus (GPa)"],
             "rows":[["FCC","3.2","180"],["BCC","5.1","210"]],
             "n_rows":2,"n_cols":3,"footnotes":[],
             "markdown_text":"| Phase | Hardness (GPa) | Modulus (GPa) |\n|---|---|---|\n| FCC | 3.2 | 180 |\n| BCC | 5.1 | 210 |"}}
```

### parsed_chunks.jsonl — 摘要行

```json
{"paper_id":"PMC11008379","chunk_id":"PMC11008379::a0000","chunk_type":"abstract",
 "text":"Pancreatic cancer remains one of the most lethal malignancies...",
 "section_path":["Abstract","BACKGROUND"],
 "metadata":{"doi":"...","title":"...","source_path":"...",
             "sec_type":null,"chunk_index":0,"token_count":95}}
```

### preprocessing_summary.json

```json
{"parser_version":"preprocessing_mvp_v1",
 "supported_format":"jats_xml_only",
 "chunking_config":{"max_chunk_tokens":400,"min_tail_tokens":200,
                    "token_upper_bound":600,"overlap_tokens":0,
                    "split_by_sentence":true},
 "total_papers":30,"parsed_ok":28,"parse_error":2,
 "total_chunks":1450,"paragraph_chunks":1300,"table_chunks":120,"abstract_chunks":30,
 "skipped_sections":{"ref-list":28,"ack":25},
 "errors":[{"paper_id":"PMCxxxx","source_path":"...","error":"no <body> found"}]}
```

---

## 8. 与现有项目结构的衔接

| 文件 | 动作 | 说明 |
|------|------|------|
| `tools/document_parser.py` | **修改** | 实现 `parse_full_text()`（现为 `NotImplementedError`）；`DocumentChunk.metadata` 放宽为 `dict[str, Any]`。facade，转发到 xml parser |
| `tools/xml_full_text_parser.py` | **新增** | JATS 全文解析核心：`walk / emit_abstract / emit_paragraph / emit_table / parse_jats_table / chunk_text / to_markdown / should_skip_section` |
| `workflow.py` | **修改** | 新增 `run_preprocess_mvp(passed_papers_path, output_dir, limit, resume, ...)`：读 passed → 续跑未完成论文 → 写 parsed_chunks.jsonl + summary。复用现有 `_write_jsonl` 风格 |
| `run_preprocess.py`（根目录） | **新增** | CLI，风格对齐 `run_paper_filter.py`：`--passed / --output / --limit / --force / --verbose` |
| `config_schema.py` | **不改** | 切分参数第一版写死常量，不进 config |
| `requirements.txt` | **修改** | 新增 `tiktoken` |
| `docs/preprocessing_mvp.md` | **本文件** | 定稿方案 |

**不改**：config_generator、llm_labeler、retriever、extractor。

### workflow 函数签名

```python
def run_preprocess_mvp(
    self,
    passed_papers_path: str | Path,   # 上一阶段 passed_papers.jsonl
    output_dir: str | Path,           # 写 parsed_chunks.jsonl + preprocessing_summary.json
    limit: int | None = None,         # 最多处理 N 篇尚未解析的论文
    resume: bool = True,              # 默认跳过已有结果；--force 时为 False
) -> dict:                            # 返回统计 counts（同 summary）
    """读 passed_papers.jsonl → 逐篇 parse_full_text → 写 chunk 流 + summary。

    只处理 file_type == "xml" 的行；非 xml 记入 errors 跳过。
    解析异常按篇捕获，不中断整体（对齐 run_paper_filter_mvp 的容错风格）。
    续跑时会保留旧 chunks，并只追加本次新解析的论文。
    """
```

### CLI 骨架（run_preprocess.py）

对齐 `run_paper_filter.py` 结构：加载 .env（可选）、`sys.path` 注入 `src/`、
`logging.basicConfig`、`_build_parser` / `main`。**本阶段不调用 LLM**，无需 client 工厂。

```
参数：
  --passed PATH   passed_papers.jsonl 路径（必填）
  --output DIR    输出目录（必填）
  --limit N       最多处理 N 篇尚未解析的论文（可选，冒烟/批处理）
  --force         重算已有解析结果
  --verbose/-v    DEBUG 日志

main 流程：
  args → 构造 DocumentParser → DomainExtractionWorkflow(document_parser=...)
       → workflow.run_preprocess_mvp(passed, output, limit)
       → 打印 total/parsed_ok/parse_error/total_chunks
```

> `--input` 不需要：passed_papers.jsonl 每行已含 `source_path`，直接据此定位 XML。

---

## 9. 模块职责边界

| 模块 | 职责 | 不负责 |
|------|------|--------|
| `xml_full_text_parser.py` | JATS 全文 → DocumentChunk；section 层级；段落/表格切分 | HTML/PDF、embedding、labeling |
| `document_parser.py` | `parse_full_text` facade；`parse_metadata_light`（已有） | 解析细节（转发给 xml parser） |
| `workflow.py` | 编排：读 passed → 逐篇解析 → 写输出 | 解析逻辑本身 |

---

## 10. 必须一致 / 必须改进（对照 ALLMAT）

**必须和 ALLMAT 一致：**
- 段落切分 token 哲学：200–600 区间、>400 才切、防碎块（`loader.py:157`）
- 表格 rowspan/colspan 网格填充算法（`loader.py:38`）
- "预处理不做领域过滤，过滤留给 labeling"分层原则（`heas/label.py:13`）

**必须比 ALLMAT 改进：**
- `chunk_type` 与 `section_path` 分离（不复用单一 `sub_titles`）
- `section_path` 用 list 而非 `/` 字符串
- 表格保留 `section_path` + 结构化 `label/caption/header_rows/headers/rows/markdown_text`（ALLMAT 全丢）
- 不写多出版社 HTML parser（`chempp` 那套整体不要）

---

## 11. 如何接到 labeling 阶段

- labeling 直接读 `parsed_chunks.jsonl`，一行一个 chunk
- **semantic search** 用 `text`（段落文本 / 表格 markdown）——对应 ALLMAT `page_content` 进 Chroma（`indexing.py:157`）
- **section filtering** 用 `section_path` + `metadata.sec_type`——对应 ALLMAT `sub_titles` 正则匹配（`heas/embeddings.py:21` `match_subtitles`），但字段更干净
- **喂 LLM** 用 `text`（表格额外可给 `caption`）——对应 ALLMAT `label.py:18` `paragraph.page_content`
- 表格与段落同一 chunk 流，labeling 可让表格一起进 semantic search，
  而非 ALLMAT 那种表格全走 regex/LLM 分类
