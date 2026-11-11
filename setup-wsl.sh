#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p work/upstream work/wsl_vendor
if [ ! -d work/upstream/GPT-Policy/.git ]; then
  git clone https://github.com/cheng-haha/GPT-Policy.git work/upstream/GPT-Policy
fi
git -C work/upstream/GPT-Policy checkout 2ca201638e1139fe35973c8a2d209a894b389db1
python3 -m pip install --target work/wsl_vendor numpy==2.2.6 Pillow==12.2.0 jsonschema rich
