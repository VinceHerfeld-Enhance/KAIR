# Source in every Abaca job (OAR + Guix modules):   source scripts/abaca/env.sh
#
# Override any of these before sourcing:
#   VENV      python venv with torch/torchvision/basicsr/wandb   (default ~/venv-py312)
#   VSR_SRC   first-party package source, put on PYTHONPATH      (default <KAIR>/../VSR/src)
# The BasicVSR++ study needs no compiled extension: the DCN is torchvision's deform_conv2d.

KAIR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KAIR_ROOT
VENV="${VENV:-$HOME/venv-py312}"
VSR_SRC="${VSR_SRC:-$KAIR_ROOT/../VSR/src}"

module load cuda-toolkit/12.9.1 2>/dev/null || true

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$VSR_SRC${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$KAIR_ROOT"

python - <<'EOF_PY' || { echo "[env] aborting"; return 1 2>/dev/null || exit 1; }
import sys, torch, torchvision
if not torch.cuda.is_available():
    sys.exit("[env] no CUDA device visible")
cc = torch.cuda.get_device_capability()
if f"sm_{cc[0]}{cc[1]}" not in torch.cuda.get_arch_list():
    sys.exit(f"[env] sm_{cc[0]}{cc[1]} ({torch.cuda.get_device_name()}) not in torch {torch.__version__} "
             f"arch list {torch.cuda.get_arch_list()}; request a supported GPU with PROPS")
import vsr.research_models.basicvsrpp_ablation  # noqa: F401  (fails fast if VSR_SRC is wrong)
print(f"[env] {torch.cuda.device_count()}x {torch.cuda.get_device_name()} sm_{cc[0]}{cc[1]}, "
      f"torch {torch.__version__}, torchvision {torchvision.__version__}")
EOF_PY
