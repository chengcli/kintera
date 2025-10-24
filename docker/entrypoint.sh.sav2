#!/usr/bin/env bash
set -e
# Activate venv for login shells and non-interactive shells
export VIRTUAL_ENV=/opt/venv
export PATH="$VIRTUAL_ENV/bin:$PATH"

# If user mounts code to /workspace, ensure ownership doesn’t break builds
# (no chown here; rely on USER_UID/GID mapping at build or run time)

# Print helpful banner
echo "Dev container ready. Python: $(python --version)"
echo "CUDA version: $(nvcc --version | sed -n 's/^.*release \(.*\),.*/\1/p')"
exec "$@"
