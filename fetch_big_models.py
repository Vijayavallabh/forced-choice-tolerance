#!/usr/bin/env python3
"""Fetch the 70B-class weights into the array, not the root disk.

The ladder needs a 70B-class open model, and "the two largest of four collect
it" has to be separable into a statement about scale or one about Qwen2.5.
Qwen2.5-72B extends the ladder this paper already has; Llama-3.3-70B
is a different family at the same scale, and Llama is the family whose 8B
member refuses the neutral framing, so it is also the test of whether that
refusal is the family or the size.

Root has 261GB free and these are ~140GB each, so both go to the data array.
No contact address is sent: the default hub user agent carries none and we do
not add one.
"""
from __future__ import annotations

import os
import sys

# The weights cache has moved once. Take the first that exists rather than a
# hardcoded path, or a re-run silently re-downloads instead of reading it. Where
# it lives on a host goes in AGENTICLS_HF_CACHES (colon-separated): a path in a
# shipped script names the machine it ran on.
if "HF_HUB_CACHE" not in os.environ:
    for _hf in os.environ.get("AGENTICLS_HF_CACHES", "").split(":"):
        if os.path.isdir(_hf):
            os.environ["HF_HUB_CACHE"] = _hf
            break
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import snapshot_download

MODELS = ["Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"]

# the weights and what is needed to tokenise; no .pth, no original/ duplicates
ALLOW = ["*.safetensors", "*.json", "*.txt", "*.model"]
IGNORE = ["original/*", "*.pth", "*.bin", "consolidated*"]


def main() -> int:
    names = sys.argv[1:] or MODELS
    for name in names:
        print(f"--- {name}", flush=True)
        try:
            path = snapshot_download(name, allow_patterns=ALLOW,
                                     ignore_patterns=IGNORE, max_workers=8)
            print(f"ok {name} -> {path}", flush=True)
        except Exception as exc:                     # noqa: BLE001
            print(f"FAILED {name}: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
