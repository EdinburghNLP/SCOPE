#!/usr/bin/env bash
# train_olmo3_7b.sh — SCOPE long-form training pipeline
# (6-GPU H100 layout, OLMo3 with function_calls_xml format).
#
# Base: training/train_qwen3_8b.sh (pipeline structure) with OLMo3 adaptations:
#   1. OLMo-3-7B-Instruct with function_calls_xml format.
#   2. Search uses <function_calls>search(query="...")</function_calls> (tool).
#   3. Answer uses <answer>...</answer> XML tags (not function_calls).
#   4. Task uses <task>...</task> XML tags (not function_calls).
#   5. Chat template: olmo3_consistent_eos.jinja (consistent <|im_end|>).
#   6. No thinking-hidden mode (OLMo3 cannot produce <think> reliably).
#   7. Dynamic user turns use chain2_xml style with format-aware final nudge.
#   8. BATCH_PER_GPU=2 (no thinking token overhead).
#
# 6-GPU layout:
#   GPUs 0-1 (SERVER_GPUS): Retrieval (FAISS) + inference servers
#   GPUs 2-5 (TRAIN_GPUS):  Training + rollout (TP=1, DP=4)
#
# Runs N iterations of:
#   Stage 1: Train Challenger (reward from latest solver; grader = base model)
#   Stage 2: Create Tasks (with trained challenger, filtered by difficulty)
#   Stage 3: Train Solver (on generated tasks; grader = base model)
#
# Recovery: use START_ITER and START_STAGE to resume from a specific point.
#   START_ITER=2 START_STAGE=3 bash training/train_olmo3_7b.sh
#
# Usage:
#   bash training/train_olmo3_7b.sh
#   NUM_ITERATIONS=5 bash training/train_olmo3_7b.sh
#   START_ITER=2 START_STAGE=3 bash training/train_olmo3_7b.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .env 2>/dev/null || true
CONDA_PREFIX="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
mkdir -p logs

# ============================================================================
# Configuration
# ============================================================================

# --- Pipeline control (overridable via environment) ---
NUM_ITERATIONS=${NUM_ITERATIONS:-3}
START_ITER=${START_ITER:-1}
START_STAGE=${START_STAGE:-1}    # 1=challenger, 2=tasks, 3=solver

# --- Model & hardware (6-GPU H100 layout) ---
BASE_MODEL="allenai/Olmo-3-7B-Instruct"
GPUS=${GPUS:-6}
SERVER_GPUS=${SERVER_GPUS:-"0,1"}        # Inference servers (retrieval + grader)
TRAIN_GPUS=${TRAIN_GPUS:-"2,3,4,5"}     # Training + rollout
TRAIN_N_GPUS=4                           # GPUs visible to trainer
SERVER_DP=2                              # Data parallel for 2-GPU servers
TP=2                                     # Tensor parallel (challenger rollout on TRAIN_GPUS)
DP=2                                     # Data parallel (challenger rollout)
SERVER_TYPE=sglang

# --- Format (OLMo3: function_calls search + XML answer/task) ---
ROLLOUT_FORMAT="function_calls_xml"
CHALLENGER_FMT="function_calls_xml"
SOLVER_FMT="function_calls_xml"
CHALLENGER_STOP='["</function_calls>","</task>"]'
SOLVER_STOP='["</function_calls>","</answer>"]'
TASK_FORMAT="function_calls_xml"
CHALLENGER_FC_TOOLS='[search]'
SOLVER_FC_TOOLS='[search]'

# --- Training hyperparameters ---
CHALLENGER_STEPS=20
SOLVER_STEPS=20
CHALLENGER_ALGORITHM=grpo
SOLVER_ALGORITHM=grpo
GRPO_GROUP_SIZE=8
REWARD_ROLLOUT_N=8
BATCH_PER_GPU=2                          # OLMo3 has no thinking tokens, can use 2
SOLVER_BATCH_SIZE=256
TRAINER_MAX_EPOCHS=1000000

# --- Templates ---
CHAT_TEMPLATE_PATH="./scope/chat_templates/olmo3_consistent_eos.jinja"
CHALLENGER_TEMPLATE="./scope/prompts/challenger_function_calls_xml.txt"
SOLVER_TEMPLATE="./scope/prompts/solver_function_calls_xml.txt"
GRADER_TEMPLATE_CHALLENGER="./scope/prompts/grader_per_rubric.txt"
GRADER_TEMPLATE_SOLVER="./scope/prompts/grader_per_rubric.txt"
RUBRIC_TEMPLATE="./scope/prompts/rubric.txt"
TASK_DESCRIPTIONS_DIR="./scope/prompts/tasks"

# --- Dynamic user turns ---
DYNAMIC_USER_TURNS=1
DYNAMIC_STYLE="chain2_xml"
MAX_ASSISTANT_TURNS=10

# --- Solver token budgets ---
SOLVER_THINK_TOKEN_BUDGET=""
SOLVER_ANSWER_TOKEN_LIMIT=2048
SOLVER_ANSWER_SOFT_LIMIT=1024
SOLVER_ANSWER_LENGTH_PENALTY_FLOOR=0.05

# --- Quality grading ---
QUALITY_GATES="entity,source_relevance"
QUALITY_ENTITY_TEMPLATE="./scope/prompts/quality_entity.txt"
QUALITY_SOURCE_RELEVANCE_TEMPLATE="./scope/prompts/quality_source_relevance.txt"
QUALITY_REQUIRED_SUM=2
QUALITY_MAX_TOKENS=256                   # OLMo3 doesn't need extra for thinking
SOLVER_PROMPT_LENGTH_LIMIT=2048          # Drop tasks exceeding solver max_prompt_length

# --- Data ---
CHALLENGER_BASE_SEED=42
CHALLENGER_VAL_DATA="./data/validation_olmo3.parquet"
SOLVER_VAL_DATA="./data/validation_olmo3.parquet"
VAL_GRADER_MODEL="gpt-5-mini"
VAL_GRADER_TEMPERATURE=1.0

# --- Config paths ---
CONFIG_PATH="./config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

# --- Memory fractions (6-GPU: servers on SERVER_GPUS, training on TRAIN_GPUS) ---
CHALLENGER_ROLLOUT_MEM_ITER1=0.30
CHALLENGER_SOLVER_MEM_ITER1=0.30
CHALLENGER_ROLLOUT_MEM=0.60       # iter2+ rollout mem on TRAIN_GPUS (no FAISS contention)
CHALLENGER_SOLVER_MEM=0.50        # iter2+ (solver is bottleneck -- needs KV cache for batching)
CHALLENGER_GRADER_MEM=0.65        # iter2+ (SGLang formula: high fraction = less reserved non-cache)

# Solver: grader on SERVER_GPUS, rollout/training on TRAIN_GPUS (no sharing)
SOLVER_ROLLOUT_MEM=0.70
SOLVER_GRADER_MEM=0.60

# --- Task creation ---
TARGET_FILTERED=5120          # 256 batch * 20 steps
PROMPT_BATCH_N=2000
MAX_TASKS_PER_PROMPT=1
TASK_MAX_ITERS=20
TASK_N_ROLLOUTS=4
TASK_REWARD_ROLLOUT_N=4
TASK_BATCH_SIZE=500
DIFFICULTY_MIN=0.2
DIFFICULTY_MAX=0.8
TASK_CHALLENGER_GPUS="0,1"
TASK_SOLVER_GPUS="2,3,4,5"
TASK_CHALLENGER_MEM=0.60
TASK_SOLVER_MEM=0.50
TASK_GRADER_MEM=0.30

# --- Server startup ---
SERVER_TIMEOUT=600

# --- Derived ---
MODEL_NAME=$(basename "$BASE_MODEL" | tr '[:upper:]' '[:lower:]')
mkdir -p logs

echo ""
echo "=================================================================="
echo "  SCOPE Training Pipeline (OLMo3 function_calls_xml, 6-GPU H100)"
echo "  Iterations: ${START_ITER}..${NUM_ITERATIONS}"
echo "  Base model: ${BASE_MODEL}"
echo "  Challenger: ${CHALLENGER_ALGORITHM}, ${CHALLENGER_STEPS} steps/iter"
echo "  Solver: ${SOLVER_ALGORITHM}, ${SOLVER_STEPS} steps/iter (batch=${SOLVER_BATCH_SIZE})"
echo "  Format: ${ROLLOUT_FORMAT}"
echo "  Server GPUs: ${SERVER_GPUS} | Train GPUs: ${TRAIN_GPUS}"
echo "  TP=${TP}, DP=${DP}, Train N_GPUs=${TRAIN_N_GPUS}"
echo "=================================================================="
echo ""

# ============================================================================
# Helper Functions
# ============================================================================

# PID tracking for reliable cleanup
RETRIEVAL_PID=""
SERVER_8001_PID=""
SERVER_8002_PID=""

start_server() {
    # Start an SGLang or VLLM inference server on specific GPUs.
    # Args: $1=gpu_list $2=port $3=model $4=dp $5=tp $6=mem $7=server_type $8=log_file
    #       [$9=served_model_name] [$10=log_level]
    local gpu_list=$1
    local port=$2
    local model=$3
    local server_dp=$4
    local server_tp=$5
    local mem=$6
    local server_type=$7
    local log_file=$8
    local served_model_name=${9:-""}
    local log_level=${10:-"info"}

    if [ "$server_type" = "vllm" ]; then
        echo "Launching VLLM server on port ${port} (GPUs: ${gpu_list}, model: $(basename $model))..."
        local cmd="CUDA_VISIBLE_DEVICES=$gpu_list python -m vllm.entrypoints.openai.api_server \
            --model=${model} \
            --port=${port} \
            --gpu-memory-utilization=${mem} \
            --tensor-parallel-size=${server_tp} \
            --max-model-len=16384"
        if [ -n "$served_model_name" ]; then
            cmd="$cmd --served-model-name=${served_model_name}"
        fi
        eval $cmd > $log_file 2>&1 &
    else
        echo "Launching SGLang server on port ${port} (GPUs: ${gpu_list}, model: $(basename $model))..."
        local cmd="CUDA_VISIBLE_DEVICES=$gpu_list python -m sglang.launch_server \
            --model=${model} \
            --port=${port} \
            --mem-fraction-static=${mem} \
            --dp-size=${server_dp} \
            --tp-size=${server_tp} \
            --log-level=${log_level}"
        if [ -n "$served_model_name" ]; then
            cmd="$cmd --served-model-name=${served_model_name}"
        fi
        eval $cmd > $log_file 2>&1 &
    fi
}

wait_for_server() {
    # Wait for server health check + inference warmup.
    # Args: $1=port $2=max_wait_seconds $3=server_type $4=model_name_for_test
    local port=$1
    local max_wait=${2:-$SERVER_TIMEOUT}
    local server_type=${3:-"sglang"}
    local model_name=${4:-""}
    local elapsed=0

    echo "Waiting for server (port ${port}) to be ready..."
    while ! curl -fsS http://127.0.0.1:${port}/health > /dev/null 2>&1; do
        if [ $elapsed -ge $max_wait ]; then
            echo "ERROR: Server on port ${port} not healthy after ${max_wait}s. Aborting."
            exit 1
        fi
        echo "  Server not ready yet, waiting 10s... (${elapsed}s elapsed)"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo "Server HTTP is up on port ${port}, testing inference..."

    if [ "$server_type" = "vllm" ]; then
        while ! curl -fsS --max-time 120 -X POST http://127.0.0.1:${port}/v1/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"'"${model_name}"'","prompt":"Hello","max_tokens":1}' > /dev/null 2>&1; do
            if [ $elapsed -ge $max_wait ]; then
                echo "ERROR: Server on port ${port} inference failed after ${max_wait}s."
                exit 1
            fi
            echo "  Not ready for inference yet, waiting 30s... (${elapsed}s)"
            sleep 30
            elapsed=$((elapsed + 30))
        done
    else
        while ! curl -fsS --max-time 120 -X POST http://127.0.0.1:${port}/generate \
            -H "Content-Type: application/json" \
            -d '{"text":"Hello","sampling_params":{"temperature":0,"max_new_tokens":1}}' > /dev/null 2>&1; do
            if [ $elapsed -ge $max_wait ]; then
                echo "ERROR: Server on port ${port} inference failed after ${max_wait}s."
                exit 1
            fi
            echo "  Not ready for inference yet, waiting 30s... (${elapsed}s)"
            sleep 30
            elapsed=$((elapsed + 30))
        done
    fi
    echo "Server on port ${port} is ready!"
}

wait_for_retrieval_server() {
    local port=$1
    local max_wait=${2:-$SERVER_TIMEOUT}
    local elapsed=0

    echo "Waiting for retrieval server (port ${port}) to be ready..."
    while ! curl -fsS http://127.0.0.1:${port}/health > /dev/null 2>&1; do
        if [ $elapsed -ge $max_wait ]; then
            echo "ERROR: Retrieval server on port ${port} not healthy after ${max_wait}s. Aborting."
            exit 1
        fi
        echo "  Retrieval server not ready yet, waiting 10s... (${elapsed}s elapsed)"
        sleep 10
        elapsed=$((elapsed + 10))
    done

    while ! curl -fsS --max-time 120 -X POST http://127.0.0.1:${port}/retrieve \
        -H "Content-Type: application/json" \
        -d '{"queries":["health check"],"topk":1,"return_scores":false}' > /dev/null 2>&1; do
        if [ $elapsed -ge $max_wait ]; then
            echo "ERROR: Retrieval server on port ${port} failed retrieval probe after ${max_wait}s. Aborting."
            exit 1
        fi
        echo "  Retrieval probe not ready yet, waiting 30s... (${elapsed}s)"
        sleep 30
        elapsed=$((elapsed + 30))
    done
    echo "Retrieval server on port ${port} is ready!"
}

cleanup_all() {
    # Kill all servers on ports 8000/8001/8002/8003 and reset PID tracking.
    echo "=== Cleaning up all servers ==="
    for port in 8000 8001 8002 8003; do
        fuser -k ${port}/tcp 2>/dev/null || true
    done
    pkill -f "sglang::data" 2>/dev/null || true
    sleep 2
    # Reset PIDs
    RETRIEVAL_PID=""
    SERVER_8001_PID=""
    SERVER_8002_PID=""
    echo "Cleanup complete."
}
trap cleanup_all EXIT INT TERM HUP

start_retrieval_server() {
    # Start the FAISS retrieval server on port 8000.
    # Args: $1=log_suffix  $2=faiss_gpu (optional, default "yes")  $3=gpu_list (optional, default $SERVER_GPUS)
    local log_suffix=$1
    local use_faiss_gpu=${2:-"yes"}
    local gpu_list=${3:-$SERVER_GPUS}
    local faiss_flag=""
    if [ "$use_faiss_gpu" = "yes" ]; then
        faiss_flag="--faiss_gpu"
        echo "Starting retrieval server (GPUs: ${gpu_list}, FAISS GPU)..."
    else
        echo "Starting retrieval server (CPU FAISS, no GPU)..."
    fi
    CUDA_VISIBLE_DEVICES=$gpu_list python search/retrieval_server.py \
        --index_path='./corpus/e5_Flat.index' \
        --corpus_path='./corpus/wiki-18.jsonl' \
        --retriever_model='intfloat/e5-base-v2' \
        --retriever_name='e5' \
        $faiss_flag \
        --topk 3 > logs/pipeline_retrieval_${log_suffix}.log 2>&1 &
    RETRIEVAL_PID=$!
}

# ============================================================================
# Checkpoint Path Functions
# ============================================================================

get_challenger_experiment_name() {
    # Returns experiment name for challenger at given iteration.
    local iter=$1
    echo "challenger_iter${iter}_${CHALLENGER_ALGORITHM}_group${GRPO_GROUP_SIZE}-${REWARD_ROLLOUT_N}_${MODEL_NAME}"
}

get_solver_experiment_name() {
    # Returns experiment name for solver at given iteration.
    local iter=$1
    echo "solver_iter${iter}_${SOLVER_ALGORITHM}_group${GRPO_GROUP_SIZE}_${MODEL_NAME}"
}

get_challenger_ckpt_path() {
    # Returns FSDP checkpoint path for challenger at given iteration.
    local iter=$1
    local step=$((iter * CHALLENGER_STEPS))
    echo "checkpoints/scope/$(get_challenger_experiment_name $iter)/global_step_${step}"
}

get_solver_ckpt_path() {
    # Returns FSDP checkpoint path for solver at given iteration.
    local iter=$1
    local step=$((iter * SOLVER_STEPS))
    echo "checkpoints/scope/$(get_solver_experiment_name $iter)/global_step_${step}"
}

get_hf_path() {
    # Returns HF-converted model path used by the training/eval scripts.
    # Input:  checkpoints/scope/exp_name/global_step_N
    # Output: checkpoints/scope/exp_name/merged_hf_actor_gsN
    local fsdp_path=$1
    local dir_name=$(basename "$fsdp_path")
    local step=${dir_name##global_step_}
    echo "$(dirname "$fsdp_path")/merged_hf_actor_gs${step}"
}

convert_fsdp_to_hf() {
    # Convert FSDP checkpoint to HuggingFace format. Prints HF path to stdout.
    # Args: $1=fsdp_checkpoint_path
    local fsdp_path=$1
    local hf_path=$(get_hf_path "$fsdp_path")

    if [ ! -d "${hf_path}" ] || [ ! -f "${hf_path}/config.json" ]; then
        echo "Converting FSDP checkpoint to HuggingFace format..." >&2
        echo "  FSDP: ${fsdp_path}" >&2
        echo "  HF:   ${hf_path}" >&2
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${fsdp_path}/actor" \
            --target_dir "${hf_path}" >&2
        if [ ! -f "${hf_path}/config.json" ]; then
            echo "ERROR: FSDP->HF conversion failed for ${fsdp_path}" >&2
            exit 1
        fi
        echo "Conversion complete: ${hf_path}" >&2
    else
        echo "HF checkpoint already exists: ${hf_path}" >&2
    fi
    echo "${hf_path}"
}

# ============================================================================
# Stage 1: Train Challenger
# ============================================================================

run_challenger() {
    local iter=$1
    local total_steps=$((iter * CHALLENGER_STEPS))
    local exp_name=$(get_challenger_experiment_name $iter)

    # Skip training if final checkpoint already exists
    local challenger_ckpt=$(get_challenger_ckpt_path $iter)
    if [ -d "$challenger_ckpt" ]; then
        echo "Challenger checkpoint already exists: $challenger_ckpt -- skipping training"
        convert_fsdp_to_hf "$challenger_ckpt" > /dev/null
        return 0
    fi

    echo ""
    echo "################################################################"
    echo "  STAGE 1: Train Challenger (iteration $iter)"
    echo "  Experiment: $exp_name"
    echo "  Total training steps: $total_steps"
    echo "################################################################"
    echo ""

    # --- Determine configuration based on iteration ---
    local rollout_mem solver_mem solver_model solver_served_name
    local grader_base_url reward_model_name
    local resume_args=()
    local extra_reward_args=()

    if [ $iter -eq 1 ]; then
        # Iter 1: solver = grader = base model, both on port 8001
        rollout_mem=$CHALLENGER_ROLLOUT_MEM_ITER1
        solver_mem=$CHALLENGER_SOLVER_MEM_ITER1
        solver_model=$BASE_MODEL
        solver_served_name=""
        grader_base_url="http://127.0.0.1:8001"
        reward_model_name=$BASE_MODEL
    else
        # Iter 2+: trained solver on 8001, base grader on 8002
        rollout_mem=$CHALLENGER_ROLLOUT_MEM
        solver_mem=$CHALLENGER_SOLVER_MEM
        solver_served_name="trained-solver"
        grader_base_url="http://127.0.0.1:8002"
        reward_model_name="trained-solver"

        # Get trained solver from previous iteration
        local prev_solver_fsdp=$(get_solver_ckpt_path $((iter - 1)))
        if [ ! -d "$prev_solver_fsdp" ]; then
            echo "ERROR: Previous solver checkpoint not found: $prev_solver_fsdp"
            exit 1
        fi
        solver_model=$(convert_fsdp_to_hf "$prev_solver_fsdp")

        # Resume from previous challenger checkpoint (FSDP resume)
        local prev_challenger_ckpt=$(get_challenger_ckpt_path $((iter - 1)))
        if [ ! -d "$prev_challenger_ckpt" ]; then
            echo "ERROR: Previous challenger checkpoint not found: $prev_challenger_ckpt"
            exit 1
        fi
        resume_args=(
            trainer.resume_mode="resume_path"
            trainer.resume_from_path="${prev_challenger_ckpt}"
        )
        extra_reward_args=(
            +custom_reward_function.reward_kwargs.grader_model_name="${BASE_MODEL}"
            +custom_reward_function.reward_kwargs.grader_server_type="${SERVER_TYPE}"
        )
    fi

    # --- Generate per-iteration challenger training data ---
    local iter_seed=$((CHALLENGER_BASE_SEED + (iter - 1) * 2 + 1))
    local iter_train_data="./data/challenger_${MODEL_NAME}_iter${iter}_seed${iter_seed}.parquet"

    if [ ! -f "$iter_train_data" ]; then
        echo "Generating challenger training data (seed=$iter_seed)..."
        python scope/process_train_challenger.py \
            --template_file $CHALLENGER_TEMPLATE \
            --task_descriptions_dir $TASK_DESCRIPTIONS_DIR \
            --num_search_turns "4:3:2" \
            --num_samples 2000 \
            --output_filename "$(basename $iter_train_data)" \
            --seed $iter_seed
        echo "Generated: $iter_train_data"
    else
        echo "Using existing challenger data: $iter_train_data"
    fi

    # --- Start servers ---
    cleanup_all

    # Start retrieval first and wait for it (FAISS GPU loads 61 GB index;
    # SGLang must start AFTER so it can accurately profile available GPU memory).
    start_retrieval_server "challenger_iter${iter}"
    wait_for_retrieval_server 8000 $SERVER_TIMEOUT

    start_server "$SERVER_GPUS" 8001 "$solver_model" $SERVER_DP 1 $solver_mem $SERVER_TYPE \
        "logs/pipeline_solver_challenger_iter${iter}.log" "$solver_served_name"
    SERVER_8001_PID=$!

    wait_for_server 8001 $SERVER_TIMEOUT $SERVER_TYPE "$reward_model_name"

    # Grader server (port 8002, iter2+ only -- always base model)
    if [ $iter -gt 1 ]; then
        start_server "$SERVER_GPUS" 8002 "$BASE_MODEL" $SERVER_DP 1 $CHALLENGER_GRADER_MEM $SERVER_TYPE \
            "logs/pipeline_grader_challenger_iter${iter}.log" ""
        SERVER_8002_PID=$!
        wait_for_server 8002 $SERVER_TIMEOUT $SERVER_TYPE "$BASE_MODEL"
    fi

    echo "All servers ready, starting challenger training on GPUs ${TRAIN_GPUS}..."

    # --- Training ---
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python -m verl.trainer.main_ppo \
        --config-path="$CONFIG_PATH" \
        --config-name='search_multiturn_grpo' \
        data.train_files=$iter_train_data \
        data.val_files=$CHALLENGER_VAL_DATA \
        data.train_batch_size=64 \
        data.shuffle=True \
        data.val_batch_size=4096 \
        data.max_prompt_length=1024 \
        data.max_response_length=8192 \
        +data.chat_template_path=${CHAT_TEMPLATE_PATH} \
        actor_rollout_ref.rollout.prompt_length=2048 \
        algorithm.use_kl_in_reward=False \
        algorithm.adv_estimator=${CHALLENGER_ALGORITHM} \
        actor_rollout_ref.model.path=${BASE_MODEL} \
        actor_rollout_ref.model.custom_chat_template=${CHAT_TEMPLATE_PATH} \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${BATCH_PER_GPU} \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.rollout.n=${GRPO_GROUP_SIZE} \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_mem} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${BATCH_PER_GPU} \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${BATCH_PER_GPU} \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=1e-3 \
        actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
        actor_rollout_ref.rollout.multi_turn.format=${ROLLOUT_FORMAT} \
        actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True \
        actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
        +actor_rollout_ref.rollout.multi_turn.dynamic_user_turns=$DYNAMIC_USER_TURNS \
        +actor_rollout_ref.rollout.multi_turn.dynamic_style=$DYNAMIC_STYLE \
        'actor_rollout_ref.rollout.multi_turn.fc_tool_names='"${CHALLENGER_FC_TOOLS}"'' \
        '+actor_rollout_ref.rollout.stop='"${CHALLENGER_STOP}"'' \
        +actor_rollout_ref.rollout.no_stop_trim=True \
        reward_model.reward_manager=batch \
        custom_reward_function.name=compute_long_challenger_score_batch \
        custom_reward_function.path=verl/custom_reward/long_reward_function_batch.py \
        custom_reward_function.reward_kwargs.model_name=${reward_model_name} \
        +custom_reward_function.reward_kwargs.solver_base_url="http://127.0.0.1:8001" \
        +custom_reward_function.reward_kwargs.grader_base_url="${grader_base_url}" \
        custom_reward_function.reward_kwargs.reward_rollout_n=${REWARD_ROLLOUT_N} \
        +custom_reward_function.reward_kwargs.solver_template_path=${SOLVER_TEMPLATE} \
        +custom_reward_function.reward_kwargs.grader_template_path=${GRADER_TEMPLATE_CHALLENGER} \
        +custom_reward_function.reward_kwargs.format_weight=0.5 \
        +custom_reward_function.reward_kwargs.difficulty_weight=1.0 \
        +custom_reward_function.reward_kwargs.grader_min_coverage=0.5 \
        +custom_reward_function.reward_kwargs.grader_pad_value=0.0 \
        +custom_reward_function.reward_kwargs.debug_print_first=True \
        +custom_reward_function.reward_kwargs.solver_server_type="${SERVER_TYPE}" \
        +custom_reward_function.reward_kwargs.retrieval_url="http://127.0.0.1:8000/retrieve" \
        +custom_reward_function.reward_kwargs.separate_rubric_generation=True \
        +custom_reward_function.reward_kwargs.rubric_template_path=${RUBRIC_TEMPLATE} \
        +custom_reward_function.reward_kwargs.tool_config_path=${TOOL_CONFIG} \
        +custom_reward_function.reward_kwargs.solver_max_prompt_length=131072 \
        +custom_reward_function.reward_kwargs.solver_timeout=3600 \
        +custom_reward_function.reward_kwargs.solver_retry=True \
        +custom_reward_function.reward_kwargs.grader_retry=True \
        +custom_reward_function.reward_kwargs.grader_format="v2" \
        '+custom_reward_function.reward_kwargs.solver_stop_tokens='"${SOLVER_STOP}"'' \
        +custom_reward_function.reward_kwargs.challenger_format="${CHALLENGER_FMT}" \
        +custom_reward_function.reward_kwargs.solver_format="${SOLVER_FMT}" \
        +custom_reward_function.reward_kwargs.tool_response_truncation_unit="token" \
        custom_reward_function.reward_kwargs.difficulty_fn="tent" \
        custom_reward_function.reward_kwargs.difficulty_target=0.5 \
        '+custom_reward_function.reward_kwargs.quality_gates="'"${QUALITY_GATES}"'"' \
        +custom_reward_function.reward_kwargs.quality_entity_template_path="${QUALITY_ENTITY_TEMPLATE}" \
        +custom_reward_function.reward_kwargs.quality_source_relevance_template_path="${QUALITY_SOURCE_RELEVANCE_TEMPLATE}" \
        +custom_reward_function.reward_kwargs.quality_required_sum="${QUALITY_REQUIRED_SUM}" \
        +custom_reward_function.reward_kwargs.quality_max_tokens="${QUALITY_MAX_TOKENS}" \
        "${extra_reward_args[@]}" \
        trainer.logger='["wandb", "console"]' \
        trainer.project_name="scope" \
        trainer.experiment_name="${exp_name}" \
        trainer.rollout_data_dir=saves/${exp_name}/rollouts \
        trainer.n_gpus_per_node=${TRAIN_N_GPUS} \
        trainer.nnodes=1 \
        trainer.save_freq=2 \
        trainer.permanent_save_freq=10 \
        trainer.max_actor_ckpt_to_keep=1 \
        trainer.test_freq=-1 \
        trainer.val_before_train=False \
        trainer.total_epochs=${TRAINER_MAX_EPOCHS} \
        trainer.total_training_steps=${total_steps} \
        "${resume_args[@]}"
    local challenger_train_status=$?
    if [ $challenger_train_status -ne 0 ]; then
        echo "ERROR: Challenger training failed (iteration $iter) with exit code $challenger_train_status"
        cleanup_all
        return $challenger_train_status
    fi

    local challenger_ckpt=$(get_challenger_ckpt_path $iter)
    if [ ! -d "$challenger_ckpt" ]; then
        echo "ERROR: Challenger trainer exited without producing target checkpoint: $challenger_ckpt"
        cleanup_all
        return 1
    fi

    echo "Challenger training complete (iteration $iter)."

    # --- Cleanup servers ---
    cleanup_all

    # --- Convert challenger FSDP->HF (needed for task creation) ---
    convert_fsdp_to_hf "$challenger_ckpt" > /dev/null
}

# ============================================================================
# Stage 2: Create Tasks
# ============================================================================

run_create_tasks() {
    local iter=$1
    local final_parquet_early="./data/gen_filtered_${MODEL_NAME}_iter${iter}.parquet"

    # Skip if final parquet already exists
    if [ -f "$final_parquet_early" ]; then
        local row_count=$(python -c "import pandas as pd; print(len(pd.read_parquet('$final_parquet_early')))")
        echo "Task parquet already exists: $final_parquet_early ($row_count rows) -- skipping creation"
        return 0
    fi

    local challenger_ckpt=$(get_challenger_ckpt_path $iter)
    local challenger_hf=$(get_hf_path "$challenger_ckpt")

    if [ ! -f "${challenger_hf}/config.json" ]; then
        echo "ERROR: Challenger HF checkpoint not found: $challenger_hf"
        exit 1
    fi

    # Determine solver model for difficulty evaluation
    local solver_model
    if [ $iter -eq 1 ]; then
        solver_model=$BASE_MODEL
    else
        local prev_solver_fsdp=$(get_solver_ckpt_path $((iter - 1)))
        solver_model=$(get_hf_path "$prev_solver_fsdp")
        if [ ! -f "${solver_model}/config.json" ]; then
            echo "ERROR: Solver HF checkpoint not found: $solver_model"
            exit 1
        fi
    fi

    local final_parquet="./data/gen_filtered_${MODEL_NAME}_iter${iter}.parquet"
    local run_dir="./data/pipeline_${MODEL_NAME}_iter${iter}"

    echo ""
    echo "################################################################"
    echo "  STAGE 2: Create Tasks (iteration $iter)"
    echo "  Challenger: $challenger_hf"
    echo "  Solver: $solver_model"
    echo "  Output: $final_parquet"
    echo "################################################################"
    echo ""

    # Compute task creation seed (sequential with challenger seed)
    local task_seed=$((CHALLENGER_BASE_SEED + (iter - 1) * 2 + 2))

    # Build common env vars for create_task.sh
    # (create_task.sh manages its own servers and cleanup trap)
    if [ "$solver_model" = "$BASE_MODEL" ]; then
        # Iter 1: solver = grader = base model, shared port
        MODEL="$challenger_hf" \
        SOLVER_MODEL="$solver_model" \
        GRADER_MODEL="$BASE_MODEL" \
        FINAL_PARQUET="$final_parquet" \
        RUN_DIR="$run_dir" \
        RETRIEVAL_GPUS="$SERVER_GPUS" \
        CHALLENGER_GPUS="$TASK_CHALLENGER_GPUS" \
        SOLVER_GPUS="$TASK_SOLVER_GPUS" \
        CHALLENGER_MEM="$TASK_CHALLENGER_MEM" \
        SOLVER_MEM="$TASK_SOLVER_MEM" \
        TARGET_FILTERED="$TARGET_FILTERED" \
        BATCH_SIZE="$TASK_BATCH_SIZE" \
        PROMPT_BATCH_N="$PROMPT_BATCH_N" \
        MAX_ITERS="$TASK_MAX_ITERS" \
        N_ROLLOUTS="$TASK_N_ROLLOUTS" \
        REWARD_ROLLOUT_N="$TASK_REWARD_ROLLOUT_N" \
        DIFFICULTY_MIN="$DIFFICULTY_MIN" \
        DIFFICULTY_MAX="$DIFFICULTY_MAX" \
        SOLVER_TEMPLATE="$SOLVER_TEMPLATE" \
        GRADER_TEMPLATE="$GRADER_TEMPLATE_SOLVER" \
        RUBRIC_TEMPLATE="$RUBRIC_TEMPLATE" \
        TOOL_CONFIG="$TOOL_CONFIG" \
        CHALLENGER_TEMPLATE="$CHALLENGER_TEMPLATE" \
        TASK_DESCRIPTIONS_DIR="$TASK_DESCRIPTIONS_DIR" \
        SERVER_TYPE="$SERVER_TYPE" \
        FORMAT="$TASK_FORMAT" \
        SEED="$task_seed" \
        QUALITY_GATES="$QUALITY_GATES" \
        QUALITY_ENTITY_TEMPLATE="$QUALITY_ENTITY_TEMPLATE" \
        QUALITY_SOURCE_RELEVANCE_TEMPLATE="$QUALITY_SOURCE_RELEVANCE_TEMPLATE" \
        QUALITY_REQUIRED_SUM="$QUALITY_REQUIRED_SUM" \
        QUALITY_MAX_TOKENS="$QUALITY_MAX_TOKENS" \
        SOLVER_PROMPT_LENGTH_LIMIT="$SOLVER_PROMPT_LENGTH_LIMIT" \
        MAX_TASKS_PER_PROMPT="$MAX_TASKS_PER_PROMPT" \
        DYNAMIC_USER_TURNS="$DYNAMIC_USER_TURNS" \
        DYNAMIC_STYLE="$DYNAMIC_STYLE" \
        MAX_ASSISTANT_TURNS="$MAX_ASSISTANT_TURNS" \
        PYTHONUNBUFFERED=1 \
        bash training/create_task.sh
    else
        # Iter 2+: separate grader needed (different model from solver)
        MODEL="$challenger_hf" \
        SOLVER_MODEL="$solver_model" \
        GRADER_MODEL="$BASE_MODEL" \
        GRADER_PORT=8003 \
        GRADER_GPUS="$TASK_CHALLENGER_GPUS" \
        GRADER_MEM="$TASK_GRADER_MEM" \
        RETRIEVAL_GPUS="$SERVER_GPUS" \
        FINAL_PARQUET="$final_parquet" \
        RUN_DIR="$run_dir" \
        CHALLENGER_GPUS="$TASK_CHALLENGER_GPUS" \
        SOLVER_GPUS="$TASK_SOLVER_GPUS" \
        CHALLENGER_MEM="$TASK_GRADER_MEM" \
        SOLVER_MEM="$TASK_SOLVER_MEM" \
        TARGET_FILTERED="$TARGET_FILTERED" \
        BATCH_SIZE="$TASK_BATCH_SIZE" \
        PROMPT_BATCH_N="$PROMPT_BATCH_N" \
        MAX_ITERS="$TASK_MAX_ITERS" \
        N_ROLLOUTS="$TASK_N_ROLLOUTS" \
        REWARD_ROLLOUT_N="$TASK_REWARD_ROLLOUT_N" \
        DIFFICULTY_MIN="$DIFFICULTY_MIN" \
        DIFFICULTY_MAX="$DIFFICULTY_MAX" \
        SOLVER_TEMPLATE="$SOLVER_TEMPLATE" \
        GRADER_TEMPLATE="$GRADER_TEMPLATE_SOLVER" \
        RUBRIC_TEMPLATE="$RUBRIC_TEMPLATE" \
        TOOL_CONFIG="$TOOL_CONFIG" \
        CHALLENGER_TEMPLATE="$CHALLENGER_TEMPLATE" \
        TASK_DESCRIPTIONS_DIR="$TASK_DESCRIPTIONS_DIR" \
        SERVER_TYPE="$SERVER_TYPE" \
        FORMAT="$TASK_FORMAT" \
        SEED="$task_seed" \
        QUALITY_GATES="$QUALITY_GATES" \
        QUALITY_ENTITY_TEMPLATE="$QUALITY_ENTITY_TEMPLATE" \
        QUALITY_SOURCE_RELEVANCE_TEMPLATE="$QUALITY_SOURCE_RELEVANCE_TEMPLATE" \
        QUALITY_REQUIRED_SUM="$QUALITY_REQUIRED_SUM" \
        QUALITY_MAX_TOKENS="$QUALITY_MAX_TOKENS" \
        SOLVER_PROMPT_LENGTH_LIMIT="$SOLVER_PROMPT_LENGTH_LIMIT" \
        MAX_TASKS_PER_PROMPT="$MAX_TASKS_PER_PROMPT" \
        DYNAMIC_USER_TURNS="$DYNAMIC_USER_TURNS" \
        DYNAMIC_STYLE="$DYNAMIC_STYLE" \
        MAX_ASSISTANT_TURNS="$MAX_ASSISTANT_TURNS" \
        PYTHONUNBUFFERED=1 \
        bash training/create_task.sh
    fi

    if [ ! -f "$final_parquet" ]; then
        echo "ERROR: Task creation did not produce output: $final_parquet"
        exit 1
    fi

    local row_count=$(python -c "import pandas as pd; print(len(pd.read_parquet('$final_parquet')))")
    echo "Task creation complete. Output: $final_parquet ($row_count rows)"
}

# ============================================================================
# Stage 3: Train Solver
# ============================================================================

run_solver() {
    local iter=$1
    local total_steps=$((iter * SOLVER_STEPS))
    local exp_name=$(get_solver_experiment_name $iter)
    local train_data="./data/gen_filtered_${MODEL_NAME}_iter${iter}.parquet"

    # Skip training if final checkpoint already exists
    local solver_ckpt=$(get_solver_ckpt_path $iter)
    if [ -d "$solver_ckpt" ]; then
        echo "Solver checkpoint already exists: $solver_ckpt -- skipping training"
        convert_fsdp_to_hf "$solver_ckpt" > /dev/null
        return 0
    fi

    # Solver uses TP=1, DP=TRAIN_N_GPUS on TRAIN_GPUS (separate from servers)
    local solver_tp=1
    local solver_dp=$TRAIN_N_GPUS

    echo ""
    echo "################################################################"
    echo "  STAGE 3: Train Solver (iteration $iter)"
    echo "  Experiment: $exp_name"
    echo "  Training data: $train_data"
    echo "  Total training steps: $total_steps (TP=${solver_tp}, DP=${solver_dp})"
    echo "  Batch size: ${SOLVER_BATCH_SIZE}"
    echo "################################################################"
    echo ""

    if [ ! -f "$train_data" ]; then
        echo "ERROR: Solver training data not found: $train_data"
        exit 1
    fi

    # Resume from previous solver checkpoint (iter2+)
    local resume_args=()
    if [ $iter -gt 1 ]; then
        local prev_solver_ckpt=$(get_solver_ckpt_path $((iter - 1)))
        if [ ! -d "$prev_solver_ckpt" ]; then
            echo "ERROR: Previous solver checkpoint not found: $prev_solver_ckpt"
            exit 1
        fi
        resume_args=(
            trainer.resume_mode="resume_path"
            trainer.resume_from_path="${prev_solver_ckpt}"
        )
    fi

    # --- Start servers ---
    cleanup_all

    # Start retrieval first and wait (FAISS GPU must finish before SGLang profiles memory).
    start_retrieval_server "solver_iter${iter}"
    wait_for_retrieval_server 8000 $SERVER_TIMEOUT

    # Grader server (port 8001, always base model, on SERVER_GPUS)
    start_server "$SERVER_GPUS" 8001 "$BASE_MODEL" $SERVER_DP $solver_tp $SOLVER_GRADER_MEM $SERVER_TYPE \
        "logs/pipeline_grader_solver_iter${iter}.log" "" "error"
    SERVER_8001_PID=$!

    wait_for_server 8001 $SERVER_TIMEOUT $SERVER_TYPE "$BASE_MODEL"

    echo "All servers ready, starting solver training on GPUs ${TRAIN_GPUS}..."

    # --- Training (on TRAIN_GPUS) ---
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python -m verl.trainer.main_ppo \
        --config-path="$CONFIG_PATH" \
        --config-name='search_multiturn_grpo' \
        data.train_files=$train_data \
        data.val_files=$SOLVER_VAL_DATA \
        data.train_batch_size=${SOLVER_BATCH_SIZE} \
        data.shuffle=True \
        data.val_batch_size=512 \
        data.max_prompt_length=2048 \
        data.max_response_length=8192 \
        +data.chat_template_path=${CHAT_TEMPLATE_PATH} \
        actor_rollout_ref.rollout.prompt_length=2048 \
        algorithm.use_kl_in_reward=False \
        algorithm.adv_estimator=${SOLVER_ALGORITHM} \
        actor_rollout_ref.model.path=${BASE_MODEL} \
        actor_rollout_ref.model.custom_chat_template=${CHAT_TEMPLATE_PATH} \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${SOLVER_BATCH_SIZE} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${BATCH_PER_GPU} \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.rollout.n=${GRPO_GROUP_SIZE} \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.gpu_memory_utilization=${SOLVER_ROLLOUT_MEM} \
        actor_rollout_ref.rollout.multi_stage_wake_up=true \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${solver_tp} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${BATCH_PER_GPU} \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${BATCH_PER_GPU} \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=1e-3 \
        actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
        actor_rollout_ref.rollout.multi_turn.format=${ROLLOUT_FORMAT} \
        actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True \
        actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
        'actor_rollout_ref.rollout.multi_turn.fc_tool_names='"${SOLVER_FC_TOOLS}"'' \
        '+actor_rollout_ref.rollout.stop='"${SOLVER_STOP}"'' \
        +actor_rollout_ref.rollout.no_stop_trim=True \
        reward_model.reward_manager=batch \
        custom_reward_function.name=compute_long_solver_score_batch \
        custom_reward_function.path=verl/custom_reward/long_solver_reward_function_batch.py \
        custom_reward_function.reward_kwargs.model_name=${BASE_MODEL} \
        +custom_reward_function.reward_kwargs.grader_base_url="http://127.0.0.1:8001" \
        +custom_reward_function.reward_kwargs.grader_template_path=${GRADER_TEMPLATE_SOLVER} \
        +custom_reward_function.reward_kwargs.grader_retry=True \
        +custom_reward_function.reward_kwargs.format_weight=0.5 \
        +custom_reward_function.reward_kwargs.acc_weight=1.0 \
        +custom_reward_function.reward_kwargs.search_weight=0.1 \
        +custom_reward_function.reward_kwargs.grader_min_coverage=0.5 \
        +custom_reward_function.reward_kwargs.grader_pad_value=0.0 \
        +custom_reward_function.reward_kwargs.debug_print_first=True \
        +custom_reward_function.reward_kwargs.retrieval_url="http://127.0.0.1:8000/retrieve" \
        '+custom_reward_function.reward_kwargs.solver_format="'"${SOLVER_FMT}"'"' \
        '+custom_reward_function.reward_kwargs.val_grader_model_name="'"${VAL_GRADER_MODEL}"'"' \
        '+custom_reward_function.reward_kwargs.val_grader_temperature='"${VAL_GRADER_TEMPERATURE}" \
        +custom_reward_function.reward_kwargs.think_token_budget=${SOLVER_THINK_TOKEN_BUDGET} \
        +custom_reward_function.reward_kwargs.answer_token_limit=${SOLVER_ANSWER_TOKEN_LIMIT} \
        +custom_reward_function.reward_kwargs.answer_soft_limit=${SOLVER_ANSWER_SOFT_LIMIT} \
        +custom_reward_function.reward_kwargs.answer_length_penalty_floor=${SOLVER_ANSWER_LENGTH_PENALTY_FLOOR} \
        trainer.logger='["wandb", "console"]' \
        trainer.project_name="scope" \
        trainer.experiment_name="${exp_name}" \
        trainer.rollout_data_dir=saves/${exp_name}/train \
        trainer.validation_data_dir=saves/${exp_name}/val \
        trainer.n_gpus_per_node=${TRAIN_N_GPUS} \
        trainer.nnodes=1 \
        trainer.save_freq=2 \
        trainer.permanent_save_freq=10 \
        trainer.max_actor_ckpt_to_keep=1 \
        trainer.test_freq=-1 \
        trainer.val_before_train=False \
        trainer.total_epochs=${TRAINER_MAX_EPOCHS} \
        trainer.total_training_steps=${total_steps} \
        "${resume_args[@]}"
    local solver_train_status=$?
    if [ $solver_train_status -ne 0 ]; then
        echo "ERROR: Solver training failed (iteration $iter) with exit code $solver_train_status"
        cleanup_all
        return $solver_train_status
    fi

    local solver_ckpt=$(get_solver_ckpt_path $iter)
    if [ ! -d "$solver_ckpt" ]; then
        echo "ERROR: Solver trainer exited without producing target checkpoint: $solver_ckpt"
        cleanup_all
        return 1
    fi

    echo "Solver training complete (iteration $iter)."

    # --- Cleanup servers ---
    cleanup_all

    # --- Convert solver FSDP->HF (needed for next iteration + standalone eval) ---
    convert_fsdp_to_hf "$solver_ckpt" > /dev/null
}

# ============================================================================
# Main Loop
# ============================================================================

for iter in $(seq $START_ITER $NUM_ITERATIONS); do
    echo ""
    echo "=================================================================="
    echo "  ITERATION $iter / $NUM_ITERATIONS"
    echo "=================================================================="

    # Determine starting stage (skip stages on first iteration for recovery)
    first_stage=1
    if [ $iter -eq $START_ITER ] && [ $START_STAGE -gt 1 ]; then
        first_stage=$START_STAGE
    fi

    # Stage 1: Train Challenger
    if [ $first_stage -le 1 ]; then
        run_challenger $iter || exit $?
    else
        echo "Skipping Stage 1 (challenger) -- resuming from stage $first_stage"
    fi

    # Stage 2: Create Tasks
    if [ $first_stage -le 2 ]; then
        run_create_tasks $iter || exit $?
    else
        echo "Skipping Stage 2 (create tasks) -- resuming from stage $first_stage"
    fi

    # Stage 3: Train Solver
    if [ $first_stage -le 3 ]; then
        run_solver $iter || exit $?
    else
        echo "Skipping Stage 3 (solver) -- resuming from stage $first_stage"
    fi
done

echo ""
echo "=================================================================="
echo "  Pipeline complete! Ran iterations ${START_ITER}..${NUM_ITERATIONS}"
echo "=================================================================="
