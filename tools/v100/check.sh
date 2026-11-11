#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1   # 始终在仓库根跑 cargo
echo "=== default (nvrtc_13 / cuda-13010) ==="
cargo check 2>&1 | tail -15
echo "default_exit=${PIPESTATUS[0]}"
echo "=== nvrtc_12 (cuda-12060) ==="
cargo check --no-default-features --features nvrtc_12 2>&1 | tail -15
echo "nvrtc12_exit=${PIPESTATUS[0]}"
