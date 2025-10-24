#!/usr/bin/sh

USAGE='./init.sh [namespace] [hf_token]'
[ $# -lt 2 ] && {
    echo Usage: $USAGE
    exit 1
}

namespace=$1
hf_token=$2

kustomization="
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: ${namespace}

resources:
  - job.yaml
  - pvc.yaml
  - svc.yaml

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
      - select:
          kind: Job
          name: vllm
        fieldPaths:
          - spec.template.spec.containers.[name=vllm-container].env.[name=HUGGING_FACE_HUB_TOKEN].valueFrom.secretKeyRef.name
          - spec.template.spec.containers.[name=infinity].env.[name=HUGGING_FACE_HUB_TOKEN].valueFrom.secretKeyRef.name
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

