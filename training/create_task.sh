#!/bin/bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Three-phase task generation pipeline with difficulty filtering.
#
# Phase 0: Create input prompts from corpus via process_train_challenger.py.
# Phase 1: Generate challenger outputs using SGLang server (OpenAI-compatible API).
# Phase 2: Evaluate difficulty via rubric gen + K solver rollouts + grading,
#           and filter to [difficulty_min, difficulty_max] range (HTTP/standalone).
#
# Uses an iterative loop with GPU-isolated co-existing servers. Challenger and
# solver servers run simultaneously on separate GPU sets, creating batches of
# prompts until TARGET_FILTERED is reached.
#
# Public environment variables:
#   MODEL            - Challenger model path (required; no default)
#   SOLVER_MODEL     - Solver model used for difficulty filtering
#   GRADER_MODEL     - Optional separate grader model
#   FINAL_PARQUET    - Final filtered output parquet path
#   RUN_DIR          - Per-iteration artifact directory
#   SEED             - Sampling seed for deterministic local shuffling
#   TARGET_FILTERED  - Number of filtered tasks to keep
#   PROMPT_BATCH_N   - Number of prompts generated per iteration
#   MAX_ITERS        - Safety cap on generation/filtering iterations
#   BATCH_SIZE       - Batch size used by difficulty filtering
#   CHALLENGER_GPUS  - CUDA_VISIBLE_DEVICES for challenger server
#   SOLVER_GPUS      - CUDA_VISIBLE_DEVICES for solver server
#   RETRIEVAL_GPUS   - Optional CUDA_VISIBLE_DEVICES for retrieval server
#   GRADER_GPUS      - Optional CUDA_VISIBLE_DEVICES for separate grader
#
# Model-specific wrapper scripts set the prompt templates, tool format,
# quality gates, dynamic-turn settings, rollout counts, and resource defaults
# used for paper reproduction. Prefer those wrappers over calling this helper
# directly unless you are debugging task generation.
#
# Example usage:
#   # Iter1: single model serves as both solver and grader
#   MODEL=checkpoints/iter1_challenger/global_step_200 bash training/create_task.sh
#
#   # Iter2: trained solver + separate base grader (grader shares challenger GPUs)
#   MODEL=checkpoints/iter2_challenger/global_step_100 \
#   SOLVER_MODEL=checkpoints/iter1_solver_hf \
#   GRADER_MODEL=Qwen/Qwen2.5-7B-Instruct \
#   GRADER_PORT=8003 \
#   CHALLENGER_MEM=0.40 GRADER_MEM=0.40 \
#   FINAL_PARQUET=./data/long_gen_filtered_iter2.parquet \
#   bash training/create_task.sh

# === Configuration ===
MODEL=${MODEL:-""}
if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL env var must be set to the challenger checkpoint path."
    exit 1
fi
N_ROLLOUTS=${N_ROLLOUTS:-4}

# Phase 1 SGLang server config
CHALLENGER_TP=${CHALLENGER_TP:-1}
CHALLENGER_MEM=${CHALLENGER_MEM:-0.60}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

SOLVER_TEMPLATE=${SOLVER_TEMPLATE:-"./scope/prompts/solver_search_r1.txt"}
TOOL_CONFIG=${TOOL_CONFIG:-"./config/search_tool_config.yaml"}
FORMAT=${FORMAT:-"search_r1"}

# Conditional stop tokens based on format
if [ "$FORMAT" = "search_r1" ]; then
    SOLVER_STOP_TOKENS=${SOLVER_STOP_TOKENS:-'["</answer>"]'}
elif [ "$FORMAT" = "function_calls" ]; then
    SOLVER_STOP_TOKENS=${SOLVER_STOP_TOKENS:-'["</function_calls>"]'}
elif [ "$FORMAT" = "function_calls_xml" ]; then
    SOLVER_STOP_TOKENS=${SOLVER_STOP_TOKENS:-'["</function_calls>","</answer>"]'}
else
    SOLVER_STOP_TOKENS=${SOLVER_STOP_TOKENS:-'["</answer>","</tool_call>"]'}
fi

# Phase 2 config
SOLVER_MODEL=${SOLVER_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
GRADER_MODEL=${GRADER_MODEL:-$SOLVER_MODEL}
SOLVER_TP=${SOLVER_TP:-1}
SOLVER_MEM=${SOLVER_MEM:-0.60}
SERVER_TYPE=${SERVER_TYPE:-"sglang"}
REWARD_ROLLOUT_N=${REWARD_ROLLOUT_N:-4}
DIFFICULTY_MIN=${DIFFICULTY_MIN:-0.2}
DIFFICULTY_MAX=${DIFFICULTY_MAX:-0.8}
BATCH_SIZE=${BATCH_SIZE:-5000}
TARGET_FILTERED=${TARGET_FILTERED:-15000}
MAX_TASKS_PER_PROMPT=${MAX_TASKS_PER_PROMPT:-0}
FINAL_PARQUET=${FINAL_PARQUET:-"./data/long_gen_filtered.parquet"}

GRADER_TEMPLATE=${GRADER_TEMPLATE:-"./scope/prompts/grader_per_rubric.txt"}
RUBRIC_TEMPLATE=${RUBRIC_TEMPLATE:-"./scope/prompts/rubric.txt"}

# Grader server config and validation are set after server GPU defaults.

# Phase 0: Prompt creation config
CORPUS_PATH=${CORPUS_PATH:-"./corpus/wiki-18.jsonl"}
INDEX_PATH=${INDEX_PATH:-"./corpus/e5_Flat.index"}
CHALLENGER_TEMPLATE=${CHALLENGER_TEMPLATE:-"./scope/prompts/challenger_search_r1.txt"}
TASK_DESCRIPTIONS_DIR=${TASK_DESCRIPTIONS_DIR:-"./scope/prompts/tasks"}
INPUT_SEARCH_TURNS=${INPUT_SEARCH_TURNS:-"4:3:2"}

# Dynamic user turns
DYNAMIC_USER_TURNS=${DYNAMIC_USER_TURNS:-0}
DYNAMIC_STYLE=${DYNAMIC_STYLE:-chain2}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-5}

# Quality filtering (opt-in via comma-separated gate names)
QUALITY_GATES=${QUALITY_GATES:-""}
QUALITY_ENTITY_TEMPLATE=${QUALITY_ENTITY_TEMPLATE:-""}
QUALITY_NO_LEAKAGE_TEMPLATE=${QUALITY_NO_LEAKAGE_TEMPLATE:-""}
QUALITY_RETRIEVAL_TEMPLATE=${QUALITY_RETRIEVAL_TEMPLATE:-""}
QUALITY_RETRIEVAL_MULTI_TEMPLATE=${QUALITY_RETRIEVAL_MULTI_TEMPLATE:-""}
QUALITY_SOURCE_RELEVANCE_TEMPLATE=${QUALITY_SOURCE_RELEVANCE_TEMPLATE:-""}
QUALITY_REQUIRED_SUM=${QUALITY_REQUIRED_SUM:--1}
QUALITY_MAX_TOKENS=${QUALITY_MAX_TOKENS:-256}
SOLVER_PROMPT_LENGTH_LIMIT=${SOLVER_PROMPT_LENGTH_LIMIT:-0}

# Iterative loop config
PROMPT_BATCH_N=${PROMPT_BATCH_N:-5000}
MAX_ITERS=${MAX_ITERS:-20}
RUN_DIR=${RUN_DIR:-"./data/iter_loop_run"}
CHALLENGER_GPUS=${CHALLENGER_GPUS:-"0,1"}
SOLVER_GPUS=${SOLVER_GPUS:-"2,3"}
CHALLENGER_PORT=${CHALLENGER_PORT:-8001}
SOLVER_PORT=${SOLVER_PORT:-8002}
RETRIEVAL_GPUS=${RETRIEVAL_GPUS:-""}

# Grader server config (defaults to challenger GPUs — grader is lightweight,
# so it shares with the challenger rather than the solver which does heavy K-rollout work)
GRADER_PORT=${GRADER_PORT:-$SOLVER_PORT}
GRADER_GPUS=${GRADER_GPUS:-$CHALLENGER_GPUS}
GRADER_MEM=${GRADER_MEM:-$SOLVER_MEM}
GRADER_TP=${GRADER_TP:-$SOLVER_TP}

# Validate: if GRADER_MODEL differs from SOLVER_MODEL, ports must differ
if [ "$GRADER_MODEL" != "$SOLVER_MODEL" ] && [ "$GRADER_PORT" = "$SOLVER_PORT" ]; then
    echo "ERROR: GRADER_MODEL ($GRADER_MODEL) differs from SOLVER_MODEL ($SOLVER_MODEL)"
    echo "but both use the same port ($SOLVER_PORT). Set GRADER_PORT to a different port."
    exit 1
fi

# NOTE: DP replicates the full model on each GPU for parallel batch inference.
# No vocab_size/attention_heads divisibility constraint (unlike TP).
# Phase 1 uses SGLang with --dp-size=$DP on port $CHALLENGER_PORT.
# SOLVER_TP constraint still applies: Qwen2.5-7B valid TP values: 1, 2, 4.

mkdir -p logs

# === Auto-convert FSDP checkpoint to HuggingFace format ===
if [ -f "$MODEL/actor/fsdp_config.json" ]; then
    CONVERTED_MODEL="${MODEL}_hf"
    if [ -f "$CONVERTED_MODEL/config.json" ]; then
        echo "FSDP checkpoint already converted at $CONVERTED_MODEL, skipping conversion."
    else
        echo "Detected FSDP checkpoint at $MODEL, converting to HuggingFace format..."
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$MODEL/actor" \
            --target_dir "$CONVERTED_MODEL"
        if [ ! -d "$CONVERTED_MODEL" ]; then
            echo "ERROR: Conversion failed - output not found: $CONVERTED_MODEL"
            exit 1
        fi
        echo "Conversion complete: $CONVERTED_MODEL"
    fi
    MODEL="$CONVERTED_MODEL"
    echo "Using converted model: $MODEL"
fi

# === Server helper functions ===

# PID tracking for reliable cleanup
RETRIEVAL_PID=""
CHALLENGER_PID=""
SOLVER_PID=""
GRADER_PID=""

start_server() {
    # Start an SGLang or VLLM server on specified GPUs and port.
    # Args: $1=gpu_list $2=port $3=model $4=dp $5=tp $6=mem $7=server_type $8=log_file
    local gpu_list=$1
    local port=$2
    local model=$3
    local dp=$4
    local tp=$5
    local mem=$6
    local server_type=$7
    local log_file=$8

    if [ "$server_type" = "vllm" ]; then
        echo "Launching VLLM server on port ${port} (GPUs: ${gpu_list})..."
        CUDA_VISIBLE_DEVICES=$gpu_list python -m vllm.entrypoints.openai.api_server \
            --model=${model} \
            --port=${port} \
            --gpu-memory-utilization=${mem} \
            --tensor-parallel-size=${tp} \
            --max-model-len=16384 > $log_file 2>&1 &
    else
        echo "Launching SGLang server on port ${port} (GPUs: ${gpu_list})..."
        local sglang_cmd="CUDA_VISIBLE_DEVICES=$gpu_list python -m sglang.launch_server \
            --model=${model} \
            --port=${port} \
            --mem-fraction-static=${mem} \
            --dp-size=${dp} \
            --tp-size=${tp} \
            --log-level=info"
        if [ "$FORMAT" != "search_r1" ] && [ "$FORMAT" != "function_calls" ] && [ "$FORMAT" != "function_calls_xml" ]; then
            sglang_cmd="$sglang_cmd --tool-call-parser=qwen25"
        fi
        eval $sglang_cmd > $log_file 2>&1 &
    fi
}

wait_for_server() {
    # Wait for server health check + inference warmup.
    # Args: $1=port $2=max_wait_seconds $3=server_type $4=model_name
    local port=$1
    local max_wait=${2:-600}
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
                echo "ERROR: Server on port ${port} not responding to inference after ${max_wait}s. Aborting."
                exit 1
            fi
            echo "  Server not ready for inference yet, waiting 30s... (${elapsed}s elapsed)"
            sleep 30
            elapsed=$((elapsed + 30))
        done
    else
        while ! curl -fsS --max-time 120 -X POST http://127.0.0.1:${port}/generate \
            -H "Content-Type: application/json" \
            -d '{"text":"Hello","sampling_params":{"temperature":0,"max_new_tokens":1}}' > /dev/null 2>&1; do
            if [ $elapsed -ge $max_wait ]; then
                echo "ERROR: Server on port ${port} not responding to inference after ${max_wait}s. Aborting."
                exit 1
            fi
            echo "  Server not ready for inference yet, waiting 30s... (${elapsed}s elapsed)"
            sleep 30
            elapsed=$((elapsed + 30))
        done
    fi
    echo "Server on port ${port} is ready!"
}

wait_for_retrieval_server() {
    local port=$1
    local max_wait=${2:-600}
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
        echo "  Retrieval probe not ready yet, waiting 30s... (${elapsed}s elapsed)"
        sleep 30
        elapsed=$((elapsed + 30))
    done
    echo "Retrieval server on port ${port} is ready!"
}

CLEANUP_DONE=0
cleanup() {
    [ "$CLEANUP_DONE" -eq 1 ] && return
    CLEANUP_DONE=1
    echo "Cleaning up servers..."
    for port in 8000 ${CHALLENGER_PORT} ${SOLVER_PORT} ${GRADER_PORT}; do
        fuser -k ${port}/tcp 2>/dev/null || true
    done
    pkill -f "sglang::data" 2>/dev/null || true
    sleep 2
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM HUP

# ===========================================================================
# Iterative batching with co-existing challenger + solver servers
# ===========================================================================
    echo ""
    echo "=========================================="
    echo "  Iterative task generation"
    echo "=========================================="
    echo "PROMPT_BATCH_N=$PROMPT_BATCH_N, TARGET_FILTERED=$TARGET_FILTERED, MAX_ITERS=$MAX_ITERS"
    echo "CHALLENGER_GPUS=$CHALLENGER_GPUS (port $CHALLENGER_PORT), SOLVER_GPUS=$SOLVER_GPUS (port $SOLVER_PORT)"
    echo "RUN_DIR=$RUN_DIR"
    echo ""

    # DP sizes derived from visible GPU counts.
    CHALLENGER_DP=$(echo $CHALLENGER_GPUS | tr ',' '\n' | wc -l | tr -d ' ')
    SOLVER_SERVER_DP=$(echo $SOLVER_GPUS | tr ',' '\n' | wc -l | tr -d ' ')
    GRADER_SERVER_DP=$(echo $GRADER_GPUS | tr ',' '\n' | wc -l | tr -d ' ')

    # Kill any existing servers on required ports
    fuser -k 8000/tcp 2>/dev/null || true
    fuser -k ${CHALLENGER_PORT}/tcp 2>/dev/null || true
    fuser -k ${SOLVER_PORT}/tcp 2>/dev/null || true
    if [ "$GRADER_PORT" != "$SOLVER_PORT" ]; then
        fuser -k ${GRADER_PORT}/tcp 2>/dev/null || true
    fi
    sleep 2

    mkdir -p "$RUN_DIR"

    # === Start all three servers ===

    # 1. Retrieval server (port 8000)
    echo "Starting retrieval server..."
    faiss_flag="--faiss_gpu"
    if [ "${FAISS_NO_GPU:-0}" = "1" ]; then
        faiss_flag=""
        echo "  (FAISS_NO_GPU=1: using CPU FAISS)"
    fi
    if [ -n "$RETRIEVAL_GPUS" ]; then
        CUDA_VISIBLE_DEVICES=$RETRIEVAL_GPUS python search/retrieval_server.py \
            --index_path="${INDEX_PATH}" \
            --corpus_path="${CORPUS_PATH}" \
            --retriever_model='intfloat/e5-base-v2' \
            --retriever_name='e5' \
            $faiss_flag \
            --topk 3 > logs/retrieval_server_create_task.log 2>&1 &
        RETRIEVAL_PID=$!
    else
        python search/retrieval_server.py \
            --index_path="${INDEX_PATH}" \
            --corpus_path="${CORPUS_PATH}" \
            --retriever_model='intfloat/e5-base-v2' \
            --retriever_name='e5' \
            $faiss_flag \
            --topk 3 > logs/retrieval_server_create_task.log 2>&1 &
        RETRIEVAL_PID=$!
    fi
    wait_for_retrieval_server 8000 600

    # 2. Challenger server (CHALLENGER_GPUS, CHALLENGER_PORT)
    start_server "$CHALLENGER_GPUS" "$CHALLENGER_PORT" "$MODEL" "$CHALLENGER_DP" "$CHALLENGER_TP" "$CHALLENGER_MEM" "sglang" "logs/challenger_server_tasks.log"
    CHALLENGER_PID=$!
    wait_for_server "$CHALLENGER_PORT" 600 "sglang" ""

    # 3. Solver server (SOLVER_GPUS, SOLVER_PORT)
    start_server "$SOLVER_GPUS" "$SOLVER_PORT" "$SOLVER_MODEL" "$SOLVER_SERVER_DP" "$SOLVER_TP" "$SOLVER_MEM" "$SERVER_TYPE" "logs/solver_server_tasks.log"
    SOLVER_PID=$!
    wait_for_server "$SOLVER_PORT" 600 "$SERVER_TYPE" "$SOLVER_MODEL"

    # 4. Grader server (separate, only when GRADER_PORT != SOLVER_PORT)
    if [ "$GRADER_PORT" != "$SOLVER_PORT" ]; then
        start_server "$GRADER_GPUS" "$GRADER_PORT" "$GRADER_MODEL" \
            "$GRADER_SERVER_DP" "$GRADER_TP" "$GRADER_MEM" "$SERVER_TYPE" \
            "logs/grader_server_tasks.log"
        GRADER_PID=$!
        wait_for_server "$GRADER_PORT" 600 "$SERVER_TYPE" "$GRADER_MODEL"
    fi

    echo ""
    echo "All servers are up and running."
    echo ""

    # === Pipelined iterative loop (Python orchestrator) ===
    # Resume logic, iterative batching, overlap of StageA(N+1) with StageB(N),
    # yield estimation, and merge are all handled by the orchestrator.
    ORCH_ARGS=(
        --model "$MODEL"
        --run-dir "$RUN_DIR"
        --target-filtered $TARGET_FILTERED
        --prompt-batch-n $PROMPT_BATCH_N
        --max-iters $MAX_ITERS
        --challenger-port $CHALLENGER_PORT
        --solver-port $SOLVER_PORT
        --grader-port $GRADER_PORT
        --solver-model "$SOLVER_MODEL"
        --grader-model "$GRADER_MODEL"
        --corpus-path "$CORPUS_PATH"
        --challenger-template "$CHALLENGER_TEMPLATE"
        --task-descriptions-dir "$TASK_DESCRIPTIONS_DIR"
        --solver-template "$SOLVER_TEMPLATE"
        --grader-template "$GRADER_TEMPLATE"
        --rubric-template "$RUBRIC_TEMPLATE"
        --tool-config "$TOOL_CONFIG"
        --format "$FORMAT"
        --n-rollouts $N_ROLLOUTS
        --reward-rollout-n $REWARD_ROLLOUT_N
        --difficulty-min $DIFFICULTY_MIN
        --difficulty-max $DIFFICULTY_MAX
        --solver-stop-tokens "$SOLVER_STOP_TOKENS"
        --batch-size $BATCH_SIZE
        --max-tasks-per-prompt $MAX_TASKS_PER_PROMPT
        --final-parquet "$FINAL_PARQUET"
        --input-search-turns "$INPUT_SEARCH_TURNS"
        --max-model-len $MAX_MODEL_LEN
        --seed ${SEED:-42}
    )
    # Quality filtering
    if [ -n "$QUALITY_GATES" ]; then
        ORCH_ARGS+=(--quality-gates "$QUALITY_GATES")
        ORCH_ARGS+=(--quality-entity-template-path "$QUALITY_ENTITY_TEMPLATE")
        ORCH_ARGS+=(--quality-no-leakage-template-path "$QUALITY_NO_LEAKAGE_TEMPLATE")
        ORCH_ARGS+=(--quality-retrieval-template-path "$QUALITY_RETRIEVAL_TEMPLATE")
        ORCH_ARGS+=(--quality-retrieval-multi-template-path "$QUALITY_RETRIEVAL_MULTI_TEMPLATE")
        ORCH_ARGS+=(--quality-source-relevance-template-path "$QUALITY_SOURCE_RELEVANCE_TEMPLATE")
        if [ "$QUALITY_REQUIRED_SUM" -ge 0 ] 2>/dev/null; then
            ORCH_ARGS+=(--quality-required-sum "$QUALITY_REQUIRED_SUM")
        fi
        if [ "$QUALITY_MAX_TOKENS" -gt 0 ] 2>/dev/null; then
            ORCH_ARGS+=(--quality-max-tokens "$QUALITY_MAX_TOKENS")
        fi
    fi
    if [ "$SOLVER_PROMPT_LENGTH_LIMIT" -gt 0 ] 2>/dev/null; then
        ORCH_ARGS+=(--solver-prompt-length-limit "$SOLVER_PROMPT_LENGTH_LIMIT")
    fi
    # Dynamic user turns
    ORCH_ARGS+=(--dynamic-style "$DYNAMIC_STYLE")
    ORCH_ARGS+=(--max-assistant-turns "$MAX_ASSISTANT_TURNS")
    if [ "$DYNAMIC_USER_TURNS" = "1" ]; then
        ORCH_ARGS+=(--dynamic-user-turns)
    fi

    python scope/pipeline_orchestrator.py "${ORCH_ARGS[@]}"
    ORCH_EXIT=$?

    # === Cleanup: kill servers ===
    cleanup

    # Clean up intermediate iteration directories after successful merge
    if [ $ORCH_EXIT -eq 0 ] && [ -f "$FINAL_PARQUET" ]; then
        echo "Cleaning up intermediate iteration directories in $RUN_DIR..."
        rm -rf "$RUN_DIR"
        echo "Cleanup complete."
    fi

    exit $ORCH_EXIT
