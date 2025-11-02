#!/usr/bin/bash

USAGE='./init.sh [config.json]'
[ $# -lt 1 ] && {
    echo Usage: $USAGE
    exit 1
}

CONFIG_JSON=$1

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet requests datasets rouge-score tiktoken huggingface_hub orjson

# Function to get all model names
get_models() {
  python3 - "$CONFIG_JSON" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
models = cfg.get("models", {})
for model_name in models.keys():
    print(model_name)
PY
}

# Function to get model-specific configuration
get_model_config() {
  local model_name="$1"
  python3 - "$CONFIG_JSON" "$model_name" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
model_name = sys.argv[2]
models = cfg.get("models", {})
model_cfg = models.get(model_name, {})
# Print model configuration values
print(cfg.get("namespace", ""))
print(cfg.get("hf_token", ""))
print(model_cfg.get("port_local", ""))
print(model_cfg.get("port_remote", ""))
print(model_cfg.get("model", model_name))  # Use model name from config or fallback to key
print(model_cfg.get("max_model_len", ""))
PY
}

# Get all models
echo "=== Loading models from config ==="
mapfile -t MODELS < <(get_models)
echo "Found ${#MODELS[@]} models: ${MODELS[@]}"

# Process each model
for model_name in "${MODELS[@]}"; do
    echo ""
    echo "=============================================="
    echo "=== Processing model: $model_name ==="
    echo "=============================================="
    echo ""
    
    # Get model-specific configuration
    mapfile -t _vals < <(get_model_config "$model_name")
    namespace="${_vals[0]:-}"
    hf_token="${_vals[1]:-}"
    port_local="${_vals[2]:-}"
    port_remote="${_vals[3]:-}"
    model="${_vals[4]:-}"
    max_model_len="${_vals[5]:-}"
    
    echo "Configuration for $model_name:"
    for var in namespace hf_token port_local port_remote model max_model_len; do
        val="${!var:-}"
        echo "  $var = $val"
        [ -z "$val" ] && {
            echo "error: $var missing for model $model_name." >&2
            exit 2
        }
    done
    
    # Create kustomization for this model
    kustomization="
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: ${namespace}

resources:
  - job.yaml
  - pvc.yaml
  - svc.yaml

configMapGenerator:
  - name: vllm-config
    literals:
      - MODEL=${model}
      - MAX_MODEL_LEN=${max_model_len}
      - VLLM_USE_FLASHINFER_SAMPLER=0
      - VLLM_LOGGING_LEVEL=DEBUG
      - VLLM_ALLOW_DEPRECATED_BEAM_SEARCH=1
      - VLLM_ATTENTION_BACKEND=TORCH_SDPA
    options:
      disableNameSuffixHash: true

secretGenerator:
  - name: hf-token-secret
    literals:
      - token=${hf_token}
    options:
      disableNameSuffixHash: true

replacements:
  - source:
      kind: Secret
      name: hf-token-secret
      fieldPath: metadata.name
    targets:
      - select: { kind: Job, name: vllm }
        fieldPaths:
          - spec.template.spec.containers.[name=vllm-container].env.[name=HUGGING_FACE_HUB_TOKEN].valueFrom.secretKeyRef.name
          - spec.template.spec.containers.[name=vllm-container].env.[name=HF_TOKEN].valueFrom.secretKeyRef.name
          - spec.template.spec.containers.[name=infinity].env.[name=HUGGING_FACE_HUB_TOKEN].valueFrom.secretKeyRef.name
          - spec.template.spec.containers.[name=infinity].env.[name=HF_TOKEN].valueFrom.secretKeyRef.name
"
    
    # Materialize kustomize dir
    mkdir -p k8s
    cp template/* k8s
    printf "%s\n" "${kustomization}" > k8s/kustomization.yaml
    
    # Always restart for new model (as requested)
    echo "Deploying model: ${model}..."
    kubectl -n "${namespace}" delete job vllm --ignore-not-found
    kubectl -n "${namespace}" delete pod -l app=vllm --ignore-not-found --force --grace-period=0 || true
    kubectl -n "${namespace}" wait --for=delete job/vllm --timeout=120s || true
    kubectl apply -k k8s
    echo "Waiting for model to be ready..."
    kubectl -n "${namespace}" wait --for=condition=Ready pod -l app=vllm --timeout=30m
    echo "Model ${model} is ready!"
    
    # Get new pod
    POD="$(kubectl -n "${namespace}" get pod -l app=vllm -o jsonpath='{.items[0].metadata.name}')"
    
    # Kill any existing port-forward
    pkill -f "port-forward.*${port_local}:${port_remote}" 2>/dev/null || true
    sleep 2
    
    # Start port forwarding
    echo "Starting port-forward on ${port_local}:${port_remote}..."
    kubectl -n "${namespace}" port-forward "pod/${POD}" "${port_local}:${port_remote}" >/dev/null 2>&1 &
    PORT_FORWARD_PID=$!
    
    # Wait for port forward to be ready
    sleep 5
    
    # Run benchmarks for this model
    echo ""
    echo "=== Running benchmarks for model: $model_name ==="
    python bench.py run "$CONFIG_JSON" --model "$model_name"
    
    # Kill port-forward for this model
    kill $PORT_FORWARD_PID 2>/dev/null || true
    
    echo ""
    echo "=== Completed benchmarks for model: $model_name ==="
    echo ""
done

echo ""
echo "=============================================="
echo "=== All models processed successfully! ==="
echo "=============================================="