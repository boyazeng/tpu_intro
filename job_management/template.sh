#!/bin/bash

# ==============================================================================
# Resilient TPU Monitoring & Relauncher
#
# This script's purpose is to:
# 1. Request a TPU through a queued resource and wait for it to become ACTIVE.
# 2. Once ACTIVE, launch a training command in the background.
# 3. Continuously monitor the state of the TPU resource.
# 4. If the TPU is preempted or fails, kill the training process, delete
#    the resource, and automatically restart the entire cycle from step 1.
# 5. The script runs indefinitely until manually terminated.
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# TODO: Update these variables for your specific project and training command.
export PROJECT_ID="my-project"
export ZONE="us-central2-b"
export TPU_VERSION="v4"
export NUM_CORES="8"
export IDENTIFIER="my-job-name"
export USER_NAME="my-name"

# --- Dynamic Variable Setup ---
export QUEUE_NAME="${USER_NAME}-${PROJECT_ID}-${TPU_VERSION}-${NUM_CORES}-${ZONE}-${IDENTIFIER}-queue"
export TPU_NAME="${USER_NAME}-${PROJECT_ID}-${TPU_VERSION}-${NUM_CORES}-${ZONE}-${IDENTIFIER}-tpu"
export CHECK_INTERVAL=60 # Seconds to wait between status checks.
training_pid="" # Initialize training_pid as empty

# Define your training command here.
export TRAINING_COMMAND="
gcloud alpha compute tpus tpu-vm ssh $USER_NAME@$TPU_NAME --ssh-key-file=~/.ssh/google_compute_engine --project=$PROJECT_ID --zone=$ZONE --worker=all \
--command 'cd my-repo/ && \
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate my-env && \
python -c \"import jax; print(jax.devices())\"'
"

# --- Everything Below This Line Does Not Need to Be Modified ---

# Determine the correct runtime version based on TPU_VERSION
if [ "$TPU_VERSION" = "v4" ]; then
    export RUNTIME_VERSION="tpu-ubuntu2204-base"
elif [ "$TPU_VERSION" = "v5litepod" ]; then
    export RUNTIME_VERSION="v2-alpha-tpuv5-lite"
elif [ "$TPU_VERSION" = "v6e" ]; then
    export RUNTIME_VERSION="v2-alpha-tpuv6e"
else
    echo "ERROR: Unsupported TPU_VERSION '$TPU_VERSION'. Exiting."
    exit 1
fi

# --- Helper Functions ---

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

get_tpu_state() {
    local state
    state=$(gcloud alpha compute tpus queued-resources describe "$QUEUE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --format="value(state.state)" 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo "NOT_FOUND"
    elif [ -z "$state" ]; then
        echo "UNKNOWN"
    else
        echo "$state"
    fi
}

create_queued_resource_with_retries() {
  local attempt=1
  while true; do
    log_message "Submitting request to create queued resource '$QUEUE_NAME' (attempt $attempt)..."
    # Run create and capture rc only; stream stdout/stderr directly
    if gcloud alpha compute tpus queued-resources create "$QUEUE_NAME" \
          --node-id="$TPU_NAME" \
          --project="$PROJECT_ID" \
          --zone="$ZONE" \
          --accelerator-type="${TPU_VERSION}-${NUM_CORES}" \
          --runtime-version="$RUNTIME_VERSION" \
          --spot \
          --provisioning-model=spot \
          --quiet; then
      log_message "Create request accepted."
      return 0
    fi

    rc=$?
    sleep 10
    ((attempt++))
  done
}

delete_queued_resource() {
    log_message "Submitting request to delete queued resource '$QUEUE_NAME'..."
    gcloud alpha compute tpus queued-resources delete "$QUEUE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --quiet --async
}

# --- Main Script Logic ---

log_message "Starting TPU lifecycle manager."
log_message "  Project: $PROJECT_ID, Zone: $ZONE, Type: ${TPU_VERSION}-${NUM_CORES}"

# This is the main recovery loop. If the training is preempted,
# the script will jump back to the start of this loop.
while true; do
    
    # === PHASE 1: PROVISIONING ===
    log_message "--- PHASE 1: PROVISIONING ---"
    
    # Loop until the resource is ACTIVE.
    while true; do
        tpu_state=$(get_tpu_state)
        log_message "Current resource state is: $tpu_state"

        case "$tpu_state" in
            "ACTIVE")
                log_message "TPU is ACTIVE. Proceeding to launch and monitor."
                break # Exit provisioning loop
                ;;
            "NOT_FOUND")
                log_message "Resource not found. Creating new resource..."
                create_queued_resource_with_retries
                sleep 30
                ;;
            "FAILED"|"SUSPENDED")
                log_message "ERROR: TPU entered a failed state ($tpu_state). Cleaning up and restarting."
                delete_queued_resource
                sleep 60 # Wait for deletion to propagate.
                ;;
            *) # CREATING, ACCEPTED, PROVISIONING, UNKNOWN, etc.
                log_message "TPU is not ready. Waiting ${CHECK_INTERVAL}s..."
                sleep "$CHECK_INTERVAL"
                ;;
        esac
    done

    # === PHASE 2: LAUNCH & MONITOR FOR PREEMPTION ===
    log_message "--- PHASE 2: LAUNCH & MONITOR ---"
    
    # Execute the training command in the background
    eval "$TRAINING_COMMAND" &
    training_pid=$!
    log_message "Training command launched with PID: $training_pid"

    # This loop ONLY monitors the TPU resource. It does not care about the training process.
    while true; do
        tpu_state=$(get_tpu_state)
        if [ "$tpu_state" == "ACTIVE" ]; then
            log_message "TPU is ACTIVE. Continuing to monitor..."
            sleep "$CHECK_INTERVAL"
        else
            # PREEMPTION DETECTED!
            log_message "WARNING: TPU is no longer ACTIVE (current state: $tpu_state). Preemption assumed."
            log_message "Killing orphaned training process (PID: $training_pid)..."
            kill -9 "$training_pid" 2>/dev/null || true
            wait "$training_pid" 2>/dev/null || true

            log_message "Cleaning up old resource and restarting the entire cycle."
            delete_queued_resource
            sleep 10
            continue 2 # Jump to the start of the main recovery loop.
        fi
    done
done
