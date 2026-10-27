#!/usr/bin/env bash
# Stream a running Kaggle kernel's logs to the terminal.
# Usage: scripts/kaggle_follow_logs.sh [owner/kernel-slug]
set -euo pipefail
KERNEL="${1:-zenitsu1agatsuma/reasonzipper-stage-a}"
echo "Following logs for ${KERNEL} (Ctrl-C to stop local follow; kernel keeps running)"
echo "Status: $(kaggle kernels status "${KERNEL}" 2>/dev/null | tail -1)"
exec kaggle kernels logs -f "${KERNEL}"
