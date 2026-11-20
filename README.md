# rocket-fuel
fine-tune a foundation model, or use ROCKET

## Setup

```
mkdir -p /data/hadar
conda create -y -p /data/hadar/envs/tor python=3.11
/data/hadar/envs/tor/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
/data/hadar/envs/tor/bin/pip install peft aeon huggingface_hub pytest
/data/hadar/envs/tor/bin/pip install -e .
/data/hadar/envs/tor/bin/python -m pytest
```

Secrets (`HF_TOKEN`, `TABPFN_TOKEN`) live only in `~/.config/tune-or-rocket/secrets.env`.
Caches go under `TOR_CACHE` (default `/data/hadar/tor-cache`, fallback `./cache`).
`TOR_PHASE` defaults to `0`.
