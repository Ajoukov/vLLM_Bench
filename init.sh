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

pyget() {
  python3 - "$CONFIG_JSON" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
# print(cfg, file=sys.stderr)
# print keys if present; else empty
print(cfg.get("namespace",""))
print(cfg.get("hf_token",""))
print(cfg.get("port_local",""))
print(cfg.get("port_remote",""))
print(cfg.get("defaults",{}).get("model",""))
print(cfg.get("defaults",{}).get("max_model_len",""))
PY
}
mapfile -t _vals < <(pyget)
namespace="${_vals[0]:-}"
hf_token="${_vals[1]:-}"
port_local="${_vals[2]:-}"
port_remote="${_vals[3]:-}"
model="${_vals[4]:-}"
max_model_len="${_vals[5]:-}"

for var in namespace hf_token port_local port_remote model max_model_len; do
    val="${!var:-}"
    echo $var = $val
    [ -z "$val" ] && {
        echo "error: $var missing. Add \"$var\" to $CONFIG_JSON." >&2
        exit 2
    }
done

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

# only restart if model changed or pod not Ready
CURRENT_MODEL="$(kubectl -n "${namespace}" get configmap vllm-config -o jsonpath='{.data.MODEL}' 2>/dev/null || echo)"
POD="$(kubectl -n "${namespace}" get pod -l app=vllm -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
READY="$( [ -n "$POD" ] && kubectl -n "${namespace}" get pod "$POD" -o jsonpath='{range .status.conditions[?(@.type=="Ready")]}{.status}{end}' 2>/dev/null || echo )"
echo $CURRENT_MODEL
echo $model
echo $READY
if [ "$CURRENT_MODEL" = "$model" ] && [ "$READY" = "True" ]; then
  echo "vLLM already running with model '${model}', skipping redeploy."
else
  echo "Deploying (model change or pod not ready)."
  kubectl -n "${namespace}" delete job vllm --ignore-not-found
  kubectl -n "${namespace}" delete pod -l app=vllm --ignore-not-found --force --grace-period=0 || true
  kubectl -n "${namespace}" wait --for=delete job/vllm --timeout=120s || true
  kubectl apply -k k8s
  echo "waiting for Ready state..."
  kubectl -n "${namespace}" wait --for=condition=Ready pod -l app=vllm --timeout=30m
  echo "bench.py is ready to go!"
fi

# get new pod
POD="$(kubectl -n "${namespace}" get pod -l app=vllm -o jsonpath='{.items[0].metadata.name}')"

kubectl -n "${namespace}" port-forward "pod/${POD}" "${port_local}:${port_remote}" >/dev/null 2>&1 &

echo "running bench.py..."
python bench.py run $CONFIG_JSON

