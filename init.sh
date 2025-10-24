#!/usr/bin/bash

USAGE='./init.sh [config.json]'
[ $# -lt 1 ] && {
    echo Usage: $USAGE
    exit 1
}

CONFIG_JSON=$1

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

# Kill existing Job (Jobs' pod template is immutable)
kubectl -n "${namespace}" delete job vllm --ignore-not-found
# In case pods linger, nuke them
kubectl -n "${namespace}" delete pod -l app=vllm --ignore-not-found --grace-period=0 --force || true
# Wait for deletion to complete (best-effort)
kubectl -n "${namespace}" wait --for=delete job/vllm --timeout=120s || true

kubectl apply -k k8s

kubectl -n "${namespace}" get pods -l app=vllm

POD="$(kubectl -n "${namespace}" get pod -l app=vllm -o jsonpath='{.items[0].metadata.name}')"

echo "waiting for Ready state..."
kubectl -n "${namespace}" wait --for=condition=Ready pod -l app=vllm --timeout=20m
echo "bench.py is ready to go!"

kubectl -n "${namespace}" port-forward "pod/${POD}" "${port_local}:${port_remote}" >/dev/null 2>&1 &

echo "running bench.py..."
./bench.py run $CONFIG_JSON

