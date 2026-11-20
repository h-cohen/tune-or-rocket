#!/bin/bash
# Recreate the Phase 0 environment on a new machine.
# Usage: ENV_PREFIX=/path/to/envs/tor TOR_CACHE=/path/to/tor-cache bash scripts/setup_env.sh
# Requires: conda, git, network access, an NVIDIA driver supporting CUDA 12.6 wheels.
# NOT yet run end-to-end on a fresh machine (written at handoff, 2026-11-20); check each step.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_PREFIX="${ENV_PREFIX:?set ENV_PREFIX, e.g. /data/<user>/envs/tor}"
TOR_CACHE="${TOR_CACHE:?set TOR_CACHE, e.g. /data/<user>/tor-cache (needs ~30 GB+)}"
PIP="$ENV_PREFIX/bin/pip"

# 1. Python 3.11 env. On hpc-node9 conda's libmamba solver was broken; the classic solver worked.
CONDA_NO_PLUGINS=true conda create -y -p "$ENV_PREFIX" python=3.11 --solver classic

# 2. torch 2.14.1+cu126 from the PyTorch index (the lock pins this exact build).
"$PIP" install "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126

# 3. Everything else from the lock, WITHOUT dependency resolution: several model packages
#    (momentfm, timesfm, toto-ts, reverso) pin old torch/transformers/numpy and were installed
#    with --no-deps on purpose. The lock is a full freeze, so all transitive deps are listed.
#    reverso is not on PyPI: it is installed from git below, so drop its lock line here.
grep -v -E '^(torch==|reverso==)' requirements.lock > /tmp/tor-lock-no-torch.txt
"$PIP" install --no-deps -r /tmp/tor-lock-no-torch.txt
"$PIP" install --no-deps "git+https://github.com/shinfxh/reverso@35f724d3ccf26527f17bc774e897ff0c07d0761a"

# 4. The harness itself.
"$PIP" install --no-deps -e .

# 5. GitHub-hosted model code + checkpoints, pinned (adapters verify commit + sha256).
#    UniShape has NO licence file: research measurement only; inclusion is the user's decision.
mkdir -p "$TOR_CACHE/unishape" "$TOR_CACHE/nutime"
[ -d "$TOR_CACHE/unishape/UniShape" ] || git clone https://github.com/qianlima-lab/UniShape.git "$TOR_CACHE/unishape/UniShape"
git -C "$TOR_CACHE/unishape/UniShape" checkout 78d326590ee959bd473c4b07ad6cbc13977c8625
[ -d "$TOR_CACHE/nutime/NuTime" ] || git clone https://github.com/chenguolin/NuTime.git "$TOR_CACHE/nutime/NuTime"
git -C "$TOR_CACHE/nutime/NuTime" checkout 9acdd3d0e82df0914217ef752fa3b95ec82cd960
# If either checkpoint is stored with git-lfs, run `git lfs pull` in that clone.

# 6. Secrets: create ~/.config/tune-or-rocket/secrets.env (mode 600) with
#    HF_TOKEN=...  and  TABPFN_TOKEN=...   — never commit it, never print it.
if [ ! -f "$HOME/.config/tune-or-rocket/secrets.env" ]; then
  echo "NOTE: create ~/.config/tune-or-rocket/secrets.env (chmod 600) with HF_TOKEN and TABPFN_TOKEN"
fi

# 7. Check.
"$ENV_PREFIX/bin/pip" check || echo "pip check complaints are expected for --no-deps packages (see HANDOFF.md)"
TOR_CACHE="$TOR_CACHE" "$ENV_PREFIX/bin/python" -c "from tor.models import available_adapters; print(available_adapters())"
echo "Then run: TOR_CACHE=$TOR_CACHE $ENV_PREFIX/bin/python -m pytest -q -m 'not slow and not network'"
