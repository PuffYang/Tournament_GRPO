#!/usr/bin/env bash
# Tournament-GRPO training entry point.
#
# Before running this script, make sure you have:
#   1. exported the API keys listed in `.env.example`
#      (WEB_SEARCH_API_KEY, JINA_API_KEY, TOURNAMENT_JUDGE_API_KEY, WANDB_API_KEY, ...);
#   2. set MODEL_PATH to a local copy of Qwen2.5-7B-Instruct (or any compatible base model);
#   3. launched a judge model (e.g. Qwen2.5-72B-Instruct) that serves an OpenAI-compatible
#      `/v1/chat/completions` endpoint, and exported TOURNAMENT_JUDGE_API_URL accordingly.
#
# Typical usage:
#   bash start.sh
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths (resolve relative to this script so the project is fully portable)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="${SCRIPT_DIR}/verl"
CONFIG_PATH="${VERL_DIR}/examples/sglang_multiturn/config"

MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH to a local copy of Qwen2.5-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-${SCRIPT_DIR}/data/train_new_train_95.parquet}"
VAL_DATA="${VAL_DATA:-${SCRIPT_DIR}/data/train_new_val_05.parquet}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompt.txt}"
DATASET_CLASS_FILE="${SCRIPT_DIR}/tournament_grpo_dataset.py"
REWARD_FILE="${SCRIPT_DIR}/tournament_grpo_reward.py"
TOOL_CONFIG="${SCRIPT_DIR}/tool_config/tournament_grpo_tool_config.yaml"
SEARCH_ENGINE="${SEARCH_ENGINE:-search_prime}"

DEEP_RESEARCH_BENCH_DIR="${SCRIPT_DIR}/deep_research_bench"
DEEP_RESEARCH_BENCH_QUERY="${DEEP_RESEARCH_BENCH_DIR}/data/prompt_data/query.jsonl"
DEEP_RESEARCH_BENCH_CRITERIA="${DEEP_RESEARCH_BENCH_DIR}/data/criteria_data/criteria.jsonl"
DEEP_RESEARCH_BENCH_REFERENCE="${DEEP_RESEARCH_BENCH_DIR}/data/test_data/cleaned_data/reference.jsonl"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ROLLOUT_DIR="${SCRIPT_DIR}/rollout_${TIMESTAMP}"
CKPT_DIR="${CKPT_DIR:-${SCRIPT_DIR}/checkpoint}"
WANDB_DIR="${WANDB_DIR:-${SCRIPT_DIR}/wandb}"
DEEP_RESEARCH_BENCH_OUTPUT_DIR="${DEEP_RESEARCH_BENCH_OUTPUT_DIR:-${SCRIPT_DIR}/deep_research_bench_ood_results}"

mkdir -p "${LOG_DIR}" "${ROLLOUT_DIR}" "${CKPT_DIR}" "${WANDB_DIR}" "${DEEP_RESEARCH_BENCH_OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export WANDB_DIR="${WANDB_DIR}"
export VLLM_USE_V1=1
export SEARCH_ENGINE="${SEARCH_ENGINE}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
if [[ -n "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
fi

# Optional HTTP(S) proxy for the Jina Reader endpoint (leave empty to disable).
export JINA_READER_PROXY="${JINA_READER_PROXY:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# Toolchain (override only if your environment ships non-default compilers).
unset CC CXX CUDAHOSTCXX NVCC_APPEND_FLAGS
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:--ccbin /usr/bin/g++}"

# ---------------------------------------------------------------------------
# API keys (NEVER hard-code real secrets here; export them in your shell or
# place them in a local `.env` file that is excluded from version control).
# ---------------------------------------------------------------------------
export WEB_SEARCH_API_URL="${WEB_SEARCH_API_URL:?Please set WEB_SEARCH_API_URL to your web search backend}"
export WEB_SEARCH_API_KEY="${WEB_SEARCH_API_KEY:?Please set WEB_SEARCH_API_KEY}"
export JINA_API_KEY="${JINA_API_KEY:-}"
export JINA_RATE_LIMIT_RPM="${JINA_RATE_LIMIT_RPM:-20}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# ---------------------------------------------------------------------------
# Tournament judge (OpenAI-compatible chat completion endpoint).
# ---------------------------------------------------------------------------
export JUDGE_HOST="${JUDGE_HOST:-localhost}"
export JUDGE_PORT="${JUDGE_PORT:-8000}"
export TOURNAMENT_JUDGE_MODEL_NAME="${TOURNAMENT_JUDGE_MODEL_NAME:-Qwen2.5-72B-Instruct}"
export TOURNAMENT_JUDGE_API_URL="${TOURNAMENT_JUDGE_API_URL:-http://${JUDGE_HOST}:${JUDGE_PORT}/v1/chat/completions}"
export TOURNAMENT_JUDGE_API_KEY="${TOURNAMENT_JUDGE_API_KEY:-EMPTY}"
export JUDGE_MODEL_NAME="${TOURNAMENT_JUDGE_MODEL_NAME}"
export JUDGE_API_URL="${TOURNAMENT_JUDGE_API_URL}"
export JUDGE_API_KEY="${TOURNAMENT_JUDGE_API_KEY}"

LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_${TIMESTAMP}.pid"

cd "${VERL_DIR}"
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"

# Choose either ["console"] or ["console","wandb"] depending on whether you
# want to log to Weights & Biases. We default to console-only so that a missing
# WANDB_API_KEY does not block reproduction.
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"

nohup python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name=search_multiturn_grpo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=18000 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    data.shuffle=False \
    data.prompt_key=prompt \
    data.custom_cls.path="${DATASET_CLASS_FILE}" \
    data.custom_cls.name=TournamentGRPOPromptDataset \
    ++data.system_prompt_file="${PROMPT_FILE}" \
    ++data.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.max_model_len=25000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=3000 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    actor_rollout_ref.rollout.multi_turn.format=tournament_grpo_xml \
    reward.num_workers=2 \
    reward.reward_manager.name=tournament \
    reward.custom_reward_function.path="${REWARD_FILE}" \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.group_size=2 \
    +reward.custom_reward_function.reward_kwargs.num_winners_per_group=1 \
    +reward.custom_reward_function.reward_kwargs.target_finalists=1 \
    +reward.custom_reward_function.reward_kwargs.score_increment=1.0 \
    +reward.custom_reward_function.reward_kwargs.equal_score_reward=0.5 \
    +reward.custom_reward_function.reward_kwargs.num_tournament_repeats=4 \
    +reward.custom_reward_function.reward_kwargs.max_tokens=1024 \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=True \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.project_name='tournament_grpo_baseline' \
    trainer.experiment_name="tournament_grpo_${TIMESTAMP}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=30 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.validation_data_source_alias=combined \
    trainer.deep_research_bench.enable=True \
    trainer.deep_research_bench.bench_dir="${DEEP_RESEARCH_BENCH_DIR}" \
    trainer.deep_research_bench.query_file="${DEEP_RESEARCH_BENCH_QUERY}" \
    trainer.deep_research_bench.criteria_file="${DEEP_RESEARCH_BENCH_CRITERIA}" \
    trainer.deep_research_bench.reference_file="${DEEP_RESEARCH_BENCH_REFERENCE}" \
    trainer.deep_research_bench.output_dir="${DEEP_RESEARCH_BENCH_OUTPUT_DIR}" \
    trainer.deep_research_bench.batch_size=32 \
    trainer.deep_research_bench.n=1 \
    trainer.deep_research_bench.do_sample=False \
    trainer.deep_research_bench.max_workers=8 \
    trainer.deep_research_bench.judge.max_tokens=8192 \
    ++trainer.rollout_data_dir="${ROLLOUT_DIR}" \
    ++trainer.rollout_log_freq=10 \
    ++trainer.rollout_log_sample_count=64 \
    ++trainer.extra_wandb_metrics='["google_search_call_count","browse_webpage_call_count","custom_reward","tournament_reward","tournament_raw_score","tournament_round_count","tournament_group_count","tournament_judge_error","format_reward","format_valid","tool_call_count","tool_output_count","google_search_cycle_count","browse_webpage_cycle_count","segment_count","stray_text_count","format_failure_reasons"]' \
    > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "CC=${CC:-<unset>}"
echo "CXX=${CXX:-<unset>}"
echo "CUDAHOSTCXX=${CUDAHOSTCXX:-<unset>}"
echo "NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-<unset>}"
echo "SEARCH_ENGINE=${SEARCH_ENGINE}"
echo "TOURNAMENT_JUDGE_MODEL_NAME=${TOURNAMENT_JUDGE_MODEL_NAME}"
echo "TOURNAMENT_JUDGE_API_URL=${TOURNAMENT_JUDGE_API_URL}"
echo "Started Tournament-GRPO training with PID $(cat "${PID_FILE}")"
echo "Log file: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo "Rollout dir: ${ROLLOUT_DIR}"
