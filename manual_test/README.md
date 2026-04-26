# Manual Test

人工测试入口。脚本已内置 live 数据环境变量，直接运行即可。

## 基础测试（NLU + 检索）

```bash
python manual_test/run_manual_query.py
```

或者直接带问题：

```bash
python manual_test/run_manual_query.py --query "中国平安最近为什么跌？"
```

## 增强测试（含分析信号展示）

```bash
python manual_test/run_retrieval_test.py
```

或者直接带问题：

```bash
python manual_test/run_retrieval_test.py --query "茅台今天股价多少"
python manual_test/run_retrieval_test.py --query "最近CPI和PMI怎么样"
python manual_test/run_retrieval_test.py --query "创业板指最近走势"
```

区别：`run_retrieval_test.py` 在终端额外打印分析信号摘要（trend/RSI/MACD/Bollinger/宏观方向等），输出文件相同。

输出目录：

- `manual_test/output/<timestamp>-<slug>/query.txt`
- `manual_test/output/<timestamp>-<slug>/nlu_result.json`
- `manual_test/output/<timestamp>-<slug>/retrieval_result.json`

说明：

- `nlu_result.json` 对应第一个模块（NLU）
- `retrieval_result.json` 对应第二个模块（数据检索），含 `analysis_summary` 字段
- 每次运行都会新建一个输出目录，不会覆盖上一次结果

## 情感分析

### 方式一：在 NLU+检索中集成（推荐）

基于现有 NLU + 检索结果运行情感分析，一步到位：

```bash
# 默认 mock 推理（无需下载模型）
python manual_test/run_manual_query.py --query "茅台最近有什么消息" --sentiment

# 使用真实 FinBERT 模型
python manual_test/run_manual_query.py --query "茅台最近有什么消息" --sentiment --real-models
```

输出目录中新增 `sentiment_results.json`。

### 方式二：独立情感分析测试

使用专门的情感分析测试脚本，支持内置 demo 数据、mock/真实模型、以及从之前 QI 运行结果加载：

```bash
# 默认模式（mock 推理，无需下载模型）
python manual_test/run_sentiment_test.py

# 使用真实 FinBERT 模型（首次运行需下载）
python manual_test/run_sentiment_test.py --real-models

# 从之前的 NLU+检索结果中加载数据
python manual_test/run_sentiment_test.py --input manual_test/output/<run-dir>

# 从自定义 JSON 文件加载数据
python manual_test/run_sentiment_test.py --json-file data.json
```

测试流程：
1. **Preprocessor** — 文本提取、语言检测、分句、实体相关句过滤
2. **SentimentClassifier** — 逐句 FinBERT 推理，多数标签聚合

输出目录：`manual_test/output/<timestamp>-sentiment-<slug>/`

输出文件：
- `nlu_input.json` — 输入的 NLU 结果
- `retrieval_input.json` — 输入的检索结果
- `preprocessed_docs.json` — 预处理后的文档
- `sentiment_results.json` — 情感分类结果
- `summary.json` — 汇总统计

## 情感分析模糊测试

```bash
# 默认 100 个随机样本
python manual_test/test_fuzz_sentiment.py

# 指定样本数和随机种子
python manual_test/test_fuzz_sentiment.py --num-samples 200 --seed 1
```

覆盖范围：
- **随机样本** — 随机 product_type、实体、文档（支持/不支持 source_type、中/英、正/负/中情感）
- **边界 NLU** — 空 entity、缺少 product_type、空 dict 等
- **边界文档** — 空列表、全部不支持类型、非字符串 body、超长文本、缺少字段
- **情感校验** — 已知极性的文本验证 label 正确性

模糊测试验证点：
- 预处理后的文档结构正确（language/sentences/entity_hits 类型正确）
- 情感结果字段在有效范围内（score/confidence ∈ [0,1]）
- 错误隔离（单文档异常不级联）
- product_type 跳过时返回空结果
- 结果无重复 evidence_id

## 情感分析功能覆盖测试

与 `test_coverage.py`（Query Intelligence 覆盖测试）对等，校验 sentiment 模块各维度的行为：

```bash
# 全部 28 个用例
python manual_test/test_coverage_sentiment.py

# 核心用例（14 条）
python manual_test/test_coverage_sentiment.py --quick
```

覆盖维度：

| 类别 | 用例数 | 验证点 |
|------|--------|--------|
| source | 6 | news / announcement / research_note / product_doc / faq / chat |
| lang | 2 | 中文 / 英文 |
| text_level | 4 | full / short / empty body / no text |
| skip | 3 | out_of_scope / product_info / trading_rule_fee |
| entity | 3 | 单实体 / 双实体 / 无实体 |
| sentiment | 5 | 正面 / 负面 / 中性（中英文）|
| batch | 4 | 全支持 / 全跳过 / 混合 / 空 |
| bounds | 1 | score/confidence 范围校验 |

测试结果保存至 `manual_test/output/coverage_sentiment_report.json`。

## 环境变量

两个测试脚本已内置以下默认值（使用 `os.environ.setdefault`，可被系统环境变量覆盖）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `QI_USE_LIVE_MARKET` | `1` | 启用 live 行情/基本面数据 |
| `QI_USE_LIVE_MACRO` | `1` | 启用 live 宏观指标 |
| `QI_USE_LIVE_NEWS` | `1` | 启用 live 新闻 |

如需覆盖（例如关闭 live 数据），在运行前设置环境变量即可：

```bash
# Windows
set QI_USE_LIVE_MARKET=0
python manual_test/run_retrieval_test.py --query "茅台今天股价多少"

# Linux/macOS
QI_USE_LIVE_MARKET=0 python manual_test/run_retrieval_test.py --query "茅台今天股价多少"
```

可选环境变量：

| 变量 | 说明 |
|---|---|
| `TUSHARE_TOKEN` | Tushare API token，有则优先使用 Tushare 行情 |
| `QI_USE_LIVE_ANNOUNCEMENT` | **禁止开启**，会导致程序 hang |

## retrieval_result.json 关键字段

下游同学重点关注 `analysis_summary`：

```
analysis_summary
├── market_signal        # 市场技术信号（个股/ETF/指数时非null）
│   ├── trend_signal     # bullish / bearish / neutral
│   ├── rsi_14           # RSI相对强弱指标
│   ├── ma5 / ma20       # 移动平均线
│   ├── macd             # MACD指标
│   ├── bollinger        # 布林带
│   ├── volatility_20d   # 20日波动率
│   └── pct_change_nd    # 多日涨跌幅
├── fundamental_signal   # 基本面信号（个股时非null）
│   ├── pe_ttm / pb / roe
│   └── valuation_assessment  # 估值评估
├── macro_signal         # 宏观信号（宏观查询时非null）
│   ├── indicators[]     # 各指标方向
│   └── overall          # expansionary / contractionary / mixed
└── data_readiness       # 数据就绪状态
    ├── has_price_data / has_fundamentals / has_macro / has_news / has_technical_indicators
    ├── relevant_intents
    └── relevant_topics
```

完整字段说明见 `docs/retrieval_output_spec.md`。
