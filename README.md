# Tournament-GRPO

This repository accompanies the EMNLP 2026 submission **"Tournament-GRPO: Tournament-style Group Relative Policy Optimization for Multi-turn Search Agents"** (anonymous submission). It contains the training entry point, the reward function, the dataset adapter, the tool definitions, and a vendored copy of [verl](https://github.com/volcengine/verl) with our additions.

> **Anonymity notice.** All author-identifying information has been removed for double-blind review. References to specific compute clusters, internal IP ranges, API keys, and corporate domains have been replaced with generic placeholders.

---

## Repository layout

```
.
├── start.sh                                 # one-click GRPO training entry point
├── prompt.txt                               # system prompt that constrains the rollout format
├── tournament_grpo_dataset.py               # dataset adapter that prepends the system prompt
├── tournament_grpo_reward.py                # tournament-style reward (format + pairwise judging)
├── tool_config/
│   └── tournament_grpo_tool_config.yaml     # GoogleSearch + BrowseWebpage tool configuration
├── data/                                    # *Public* training / validation splits (Parquet)
├── deep_research_bench/                     # OOD evaluation suite (RACE / FACT)
│   ├── data/                                # public bench queries, criteria, references
│   └── ood_race_eval.py                     # OOD RACE evaluator
└── verl/                                    # vendored verl with our additions
    └── verl/tools/tournament_grpo_search_tools.py  # GoogleSearchTool / BrowseWebpageTool
```

The `verl/` directory is based on [verl](https://github.com/volcengine/verl) at commit **`68227b90`**, with the following additions / modifications:

| File | Status | Purpose |
|---|---|---|
| `verl/verl/tools/tournament_grpo_search_tools.py` | **new** | Web search & web browsing tools used by the agent |
| `verl/verl/experimental/agent_loop/tool_parser.py` | modified | Registers `tournament_grpo_xml` parser |
| `verl/verl/experimental/agent_loop/tool_agent_loop.py` | modified | Inline-XML tool protocol routing |
| `verl/tests/experimental/agent_loop/test_tournament_grpo_xml_tool_parser_on_cpu.py` | **new** | Unit tests for the XML parser |
| `verl/tests/experimental/reward_loop/test_tournament_grpo_reward_on_cpu.py` | **new** | Unit tests for the reward function |
| `verl/tests/experimental/agent_loop/test_agent_loop_extra_fields_schema_on_cpu.py` | modified | Adds `tournament_grpo_xml` test cases |

The rest of `verl/` is unmodified upstream code.

---

## Requirements

* Python 3.10+
* CUDA 12.x with PyTorch 2.6/2.7 (refer to `verl/requirements.txt`)
* vLLM (for the rollout engine) or SGLang
* A locally hosted base model (we use `Qwen2.5-7B-Instruct`)
* A locally hosted judge model exposed via an OpenAI-compatible `/v1/chat/completions` endpoint (we use `Qwen2.5-72B-Instruct`)
* Access to a web search backend (any backend that returns `search_result: [{title, url, snippet, ...}]`) and optionally Jina Reader (https://r.jina.ai) for webpage extraction

Install verl and its extras:

```bash
cd verl
pip install -e ".[vllm]"   # or ".[sglang]"
cd ..
```

---

## Reproducing the training run

1. **Prepare a `.env`** (or simply `export` the variables) by copying `.env.example` and filling in your own values:

   ```bash
   cp .env.example .env
   $EDITOR .env
   set -a && source .env && set +a
   ```

2. **Launch the judge model.** Any OpenAI-compatible inference server that serves `Qwen2.5-72B-Instruct` (vLLM / SGLang / TGI) works. Then point `TOURNAMENT_JUDGE_API_URL` at it.

3. **Run training:**

   ```bash
   bash start.sh
   ```

   Logs, PID files, and rollout dumps will be written under `./logs/`, `./rollout_<timestamp>/`, `./checkpoint/`, and `./wandb/`.

### Key knobs (all overridable via environment variables)

| Env var | Default | Description |
|---|---|---|
| `MODEL_PATH` | *(required)* | Local path to `Qwen2.5-7B-Instruct` |
| `TRAIN_DATA` / `VAL_DATA` | `data/train_new_train_95.parquet` / `data/train_new_val_05.parquet` | Public training / validation splits |
| `WEB_SEARCH_API_URL` | *(required)* | Endpoint that returns search results |
| `WEB_SEARCH_API_KEY` | *(required)* | API key for the web search backend |
| `JINA_API_KEY` | `""` | Optional bearer token for `r.jina.ai` |
| `TOURNAMENT_JUDGE_API_URL` | `http://localhost:8000/v1/chat/completions` | OpenAI-compatible judge endpoint |
| `TOURNAMENT_JUDGE_API_KEY` | `EMPTY` | Bearer token for the judge endpoint |
| `TOURNAMENT_JUDGE_MODEL_NAME` | `Qwen2.5-72B-Instruct` | Model name passed to the judge |
| `N_GPUS_PER_NODE` | `8` | GPUs per node |
| `TRAINER_LOGGER` | `["console"]` | Set to `["console","wandb"]` to enable W&B |

---

## Data

`data/train_new_train_95.parquet` and `data/train_new_val_05.parquet` are derived from **public datasets**. The schema is:

| Column | Type | Description |
|---|---|---|
| `prompt` | `list[dict]` (chat-formatted) | User query |
| `ground_truth` | `dict` | `{"query": ..., "rubrics": [...]}` used by the tournament judge |
| `data_source` | `str` | Source dataset name |

See the comments in `tournament_grpo_dataset.py` for how the system prompt in `prompt.txt` is prepended at load time.

---

## Citation

To preserve double-blind review, the citation is intentionally omitted. It will be added in the camera-ready version.

---

## License

The vendored `verl/` directory retains its original Apache-2.0 license (see `verl/LICENSE`). All code we add in this repository is released under the Apache-2.0 license as well (see `LICENSE`).
