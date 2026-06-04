#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="dynhamr"

# Find conda env path
CONDA_BASE="$(conda info --base)"
ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"

if [ ! -d "$ENV_PREFIX" ]; then
    echo "ERROR: conda env not found: $ENV_PREFIX"
    echo "Please create/activate env first, e.g.: conda create -n dynhamr python=3.10"
    exit 1
fi

ACTIVATE_DIR="$ENV_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$ENV_PREFIX/etc/conda/deactivate.d"

mkdir -p "$ACTIVATE_DIR"
mkdir -p "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/env_paths.sh" <<'EOF'
#!/usr/bin/env bash

# Save old values
export OLD_DYNHAMR_PATH="${PATH:-}"
export OLD_DYNHAMR_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export OLD_DYNHAMR_CPATH="${CPATH:-}"
export OLD_DYNHAMR_CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}"
export OLD_DYNHAMR_CUDA_HOME="${CUDA_HOME:-}"
export OLD_DYNHAMR_CUDA_PATH="${CUDA_PATH:-}"

# CUDA from current conda env
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"

# Executables
export PATH="$CONDA_PREFIX/bin:$PATH"

# Libraries: conda libs + CUDA target libs
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

# Headers: conda includes + Eigen + CUDA + CCCL/Thrust
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/include/eigen3:$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/targets/x86_64-linux/include/cccl:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:$CONDA_PREFIX/include/eigen3:$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/targets/x86_64-linux/include/cccl:${CPLUS_INCLUDE_PATH:-}"

# Optional: system Eigen path if installed by apt
if [ -d "/usr/include/eigen3" ]; then
    export CPATH="/usr/include/eigen3:$CPATH"
    export CPLUS_INCLUDE_PATH="/usr/include/eigen3:$CPLUS_INCLUDE_PATH"
fi
EOF

cat > "$DEACTIVATE_DIR/env_paths.sh" <<'EOF'
#!/usr/bin/env bash

# Restore PATH
if [ -n "${OLD_DYNHAMR_PATH:-}" ]; then
    export PATH="$OLD_DYNHAMR_PATH"
fi

# Restore LD_LIBRARY_PATH
if [ -n "${OLD_DYNHAMR_LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$OLD_DYNHAMR_LD_LIBRARY_PATH"
else
    unset LD_LIBRARY_PATH
fi

# Restore CPATH
if [ -n "${OLD_DYNHAMR_CPATH:-}" ]; then
    export CPATH="$OLD_DYNHAMR_CPATH"
else
    unset CPATH
fi

# Restore CPLUS_INCLUDE_PATH
if [ -n "${OLD_DYNHAMR_CPLUS_INCLUDE_PATH:-}" ]; then
    export CPLUS_INCLUDE_PATH="$OLD_DYNHAMR_CPLUS_INCLUDE_PATH"
else
    unset CPLUS_INCLUDE_PATH
fi

# Restore CUDA_HOME
if [ -n "${OLD_DYNHAMR_CUDA_HOME:-}" ]; then
    export CUDA_HOME="$OLD_DYNHAMR_CUDA_HOME"
else
    unset CUDA_HOME
fi

# Restore CUDA_PATH
if [ -n "${OLD_DYNHAMR_CUDA_PATH:-}" ]; then
    export CUDA_PATH="$OLD_DYNHAMR_CUDA_PATH"
else
    unset CUDA_PATH
fi

# Clean backup variables
unset OLD_DYNHAMR_PATH
unset OLD_DYNHAMR_LD_LIBRARY_PATH
unset OLD_DYNHAMR_CPATH
unset OLD_DYNHAMR_CPLUS_INCLUDE_PATH
unset OLD_DYNHAMR_CUDA_HOME
unset OLD_DYNHAMR_CUDA_PATH
EOF

chmod +x "$ACTIVATE_DIR/env_paths.sh"
chmod +x "$DEACTIVATE_DIR/env_paths.sh"

echo "Done."
echo "Created:"
echo "  $ACTIVATE_DIR/env_paths.sh"
echo "  $DEACTIVATE_DIR/env_paths.sh"
echo
echo "Now run:"
echo "  conda deactivate"
echo "  conda activate dynhamr"
echo
echo "Then check:"
echo "  echo \$CUDA_HOME"
echo "  which nvcc"
echo "  echo \$CPATH | tr ':' '\\n'"
echo "  echo \$LD_LIBRARY_PATH | tr ':' '\\n'"
