#!/usr/bin/bash

USAGE='./init.sh [config.json] | ./init.sh --cleanup [config.json]'

# Check for cleanup flag
if [ "$1" = "--cleanup" ]; then
    [ $# -lt 2 ] && {
        echo "Usage: $USAGE"
        exit 1
    }
    CONFIG_JSON=$2
    
    echo "=============================================="
    echo "=== Starting cleanup process ==="
    echo "=============================================="
    echo ""
    
    # Function to get all model namespaces
    get_namespaces() {
      python3 - "$CONFIG_JSON" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
namespace = cfg.get("namespace", "")
if namespace:
    print(namespace)
PY
    }
    
    # Kill all port-forward processes
    echo "Killing all port-forward processes..."
    pkill -f "kubectl.*port-forward" 2>/dev/null || true
    
    # Get namespace from config
    namespace=$(get_namespaces)
    
    if [ -n "$namespace" ]; then
        echo "Cleaning up Kubernetes resources in namespace: $namespace"
        
        # Delete jobs
        echo "  - Deleting jobs..."
        kubectl -n "${namespace}" delete job vllm --ignore-not-found
        
        # Delete pods forcefully
        echo "  - Deleting pods..."
        kubectl -n "${namespace}" delete pod -l app=vllm --ignore-not-found --force --grace-period=0 || true
        
        # Delete services
        echo "  - Deleting services..."
        kubectl -n "${namespace}" delete svc -l app=vllm --ignore-not-found
        
        # Delete PVCs (optional, comment out if you want to keep them)
        echo "  - Deleting PVCs..."
        kubectl -n "${namespace}" delete pvc -l app=vllm --ignore-not-found
        
        # Delete configmaps and secrets
        echo "  - Deleting configmaps and secrets..."
        kubectl -n "${namespace}" delete configmap vllm-config --ignore-not-found
        kubectl -n "${namespace}" delete secret hf-token-secret --ignore-not-found
        
        echo "Kubernetes resources cleaned up successfully!"
    else
        echo "Warning: Could not determine namespace from config"
    fi
    
    # Clean up k8s directory
    if [ -d "k8s" ]; then
        echo "Removing k8s directory..."
        rm -rf k8s
    fi
    
    # Clean up Python cache directories
    echo "Cleaning up Python cache directories..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    
    # Clean up Hugging Face cache
    if [ -d ".hf-cache" ]; then
        echo "Removing Hugging Face cache (.hf-cache)..."
        rm -rf .hf-cache
    fi
    
    # Also check for default HF cache location
    if [ -d "$HOME/.cache/huggingface" ]; then
        echo "Found Hugging Face cache at $HOME/.cache/huggingface"
        read -p "Do you want to delete it? This may be shared with other projects. (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing $HOME/.cache/huggingface..."
            rm -rf "$HOME/.cache/huggingface"
        else
            echo "Skipping $HOME/.cache/huggingface"
        fi
    fi
    
    # Clean up virtual environment
    if [ -d ".venv" ]; then
        echo "Removing Python virtual environment..."
        rm -rf .venv
    fi
    
    echo ""
    echo "=============================================="
    echo "=== Cleanup completed successfully! ==="
    echo "=============================================="
    
    exit 0
fi

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
lmcache_cfg = model_cfg.get("lmcache", {})
# Print model configuration values
print(cfg.get("namespace", ""))
print(cfg.get("hf_token", ""))
print(model_cfg.get("port_local", ""))
print(model_cfg.get("port_remote", ""))
print(model_cfg.get("model", model_name))  # Use model name from config or fallback to key
print(model_cfg.get("max_model_len", ""))
# LMCache configuration - basic
print(str(lmcache_cfg.get("enabled", False)).lower())
print(lmcache_cfg.get("chunk_size", 256))
# LMCache local CPU backend
print(str(lmcache_cfg.get("local_cpu", True)).lower())
print(lmcache_cfg.get("max_local_cpu_size", 20))
# LMCache local disk backend (conditional)
local_disk = lmcache_cfg.get("local_disk", False)
print(str(local_disk).lower())
print(lmcache_cfg.get("local_disk_path", "") if local_disk else "")
print(lmcache_cfg.get("max_local_disk_size", 100) if local_disk else 0)
# LMCache remote backend (conditional)
remote_url = lmcache_cfg.get("remote_url", "")
print(remote_url)
print(lmcache_cfg.get("remote_serde", "safetensors") if remote_url else "")
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
    lmcache_enabled="${_vals[6]:-false}"
    lmcache_chunk_size="${_vals[7]:-256}"
    lmcache_local_cpu="${_vals[8]:-true}"
    lmcache_max_local_cpu_size="${_vals[9]:-20}"
    lmcache_local_disk="${_vals[10]:-false}"
    lmcache_local_disk_path="${_vals[11]:-}"
    lmcache_max_local_disk_size="${_vals[12]:-0}"
    lmcache_remote_url="${_vals[13]:-}"
    lmcache_remote_serde="${_vals[14]:-}"
    
    echo "Configuration for $model_name:"
    for var in namespace hf_token port_local port_remote model max_model_len; do
        val="${!var:-}"
        echo "  $var = $val"
        [ -z "$val" ] && {
            echo "error: $var missing for model $model_name." >&2
            exit 2
        }
    done
    
    if [ "$lmcache_enabled" = "true" ]; then
        echo "LMCache Configuration:"
        echo "  enabled = true"
        echo "  chunk_size = $lmcache_chunk_size"
        echo "  local_cpu = $lmcache_local_cpu (max_size: ${lmcache_max_local_cpu_size}GB)"
        [ "$lmcache_local_disk" = "true" ] && echo "  local_disk = true (path: ${lmcache_local_disk_path}, max_size: ${lmcache_max_local_disk_size}GB)"
        [ -n "$lmcache_remote_url" ] && echo "  remote_url = $lmcache_remote_url (serde: ${lmcache_remote_serde})"
    else
        echo "LMCache: disabled"
    fi
    
    # Create kustomization for this model
    # Build LMCache config literals conditionally
    lmcache_literals=""
    if [ "$lmcache_enabled" = "true" ]; then
        lmcache_literals="
      - LMCACHE_ENABLED=true
      - LMCACHE_CHUNK_SIZE=${lmcache_chunk_size}
      - LMCACHE_LOCAL_CPU=${lmcache_local_cpu}
      - LMCACHE_MAX_LOCAL_CPU_SIZE=${lmcache_max_local_cpu_size}"
        
        if [ "$lmcache_local_disk" = "true" ] && [ -n "$lmcache_local_disk_path" ]; then
            lmcache_literals="${lmcache_literals}
      - LMCACHE_LOCAL_DISK=true
      - LMCACHE_LOCAL_DISK_PATH=${lmcache_local_disk_path}
      - LMCACHE_MAX_LOCAL_DISK_SIZE=${lmcache_max_local_disk_size}"
        fi
        
        if [ -n "$lmcache_remote_url" ]; then
            lmcache_literals="${lmcache_literals}
      - LMCACHE_REMOTE_URL=${lmcache_remote_url}"
            [ -n "$lmcache_remote_serde" ] && lmcache_literals="${lmcache_literals}
      - LMCACHE_REMOTE_SERDE=${lmcache_remote_serde}"
        fi
    else
        lmcache_literals="
      - LMCACHE_ENABLED=false"
    fi
    
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
      - VLLM_ATTENTION_BACKEND=TORCH_SDPA${lmcache_literals}
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