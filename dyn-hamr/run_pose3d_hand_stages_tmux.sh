#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SESSION_NAME="${SESSION_NAME:-pose3d_hand_stages}"
CONDA_ENV="${CONDA_ENV:-dynhamr}"
SEQ="${SEQ:-segment_000_ch1_undistort}"
STAGE="${STAGE:-all}"

POSE3D_HAND="${POSE3D_HAND:-../demo/${SEQ}.compat.pose3d_hand}"
TRACK_INFO="${TRACK_INFO:-../demo/${SEQ}_track_info.compat.npy}"
KEYPOINTS_NPY="${KEYPOINTS_NPY:-../demo/${SEQ}_keypoints.npy}"
IMAGE_ROOT="${IMAGE_ROOT:-../demo/${SEQ}_frames}"
VITPOSE_CONFIG="${VITPOSE_CONFIG:-../third-party/hamer/third-party/ViTPose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py}"
VITPOSE_CHECKPOINT="${VITPOSE_CHECKPOINT:-../_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth}"
EXTRACT_KEYPOINTS="${EXTRACT_KEYPOINTS:-true}"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-}"
WORK_DIR="${WORK_DIR:-../outputs/pose3d_hand_stages_2/${SEQ}}"
OUTPUT="${OUTPUT:-../demo/${SEQ}_export.pose3d_hand}"
LOG_FILE="${LOG_FILE:-${WORK_DIR}/run_${STAGE}_$(date +%Y%m%d_%H%M%S).log}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RESUME="${RESUME:-True}"
SAVE_LOSS_PLOTS="${SAVE_LOSS_PLOTS:-true}"

if [[ "${LOG_FILE}" = /* ]]; then
  LOG_PATH="${LOG_FILE}"
else
  LOG_PATH="${SCRIPT_DIR}/${LOG_FILE}"
fi
LOG_DIR="$(dirname "${LOG_PATH}")"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH" >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  echo "查看: tmux attach -t ${SESSION_NAME}" >&2
  echo "终止: tmux kill-session -t ${SESSION_NAME}" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  CONDA_BASE="${CONDA_BASE:-}"
fi

if [[ -z "${CONDA_BASE}" || ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "Cannot locate conda.sh. Set CONDA_BASE or ensure conda is on PATH." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

runner_cmd=(
  python run_pose3d_hand_stages.py
  data=video_new
  "data.stage=${STAGE}"
  "data.seq=${SEQ}"
  "data.pose3d_hand=${POSE3D_HAND}"
  "data.track_info=${TRACK_INFO}"
  "data.keypoints_npy=${KEYPOINTS_NPY}"
  "data.extract_keypoints=${EXTRACT_KEYPOINTS}"
  "data.image_root=${IMAGE_ROOT}"
  "data.vitpose_config=${VITPOSE_CONFIG}"
  "data.vitpose_checkpoint=${VITPOSE_CHECKPOINT}"
  "data.work_dir=${WORK_DIR}"
  "data.output=${OUTPUT}"
  "data.resume=${RESUME}"
  "data.save_loss_plots=${SAVE_LOSS_PLOTS}"
)

if [[ -n "${DEVICE_OVERRIDE}" ]]; then
  runner_cmd+=("data.device_override=${DEVICE_OVERRIDE}")
fi

runner_cmd_str="$(printf "%q " "${runner_cmd[@]}")"
if [[ -n "${EXTRA_ARGS}" ]]; then
  runner_cmd_str+="${EXTRA_ARGS}"
fi

tmux_command=$(cat <<EOF
set -euo pipefail
cd $(printf "%q" "${SCRIPT_DIR}")
source $(printf "%q" "${CONDA_BASE}/etc/profile.d/conda.sh")
conda activate $(printf "%q" "${CONDA_ENV}")
mkdir -p $(printf "%q" "${LOG_DIR}")
echo "[$(date '+%Y-%m-%d %H:%M:%S')] session=${SESSION_NAME}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] log=${LOG_PATH}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] command=${runner_cmd_str}"
${runner_cmd_str} 2>&1 | tee -a $(printf "%q" "${LOG_PATH}")
EOF
)

tmux new-session -d -s "${SESSION_NAME}" "${tmux_command}"

echo "tmux session 已启动: ${SESSION_NAME}"
echo "日志文件: ${LOG_PATH}"
echo
echo "查看 tmux 控制台:"
echo "  tmux attach -t ${SESSION_NAME}"
echo
echo "查看实时日志:"
echo "  tail -f ${LOG_PATH}"
echo
echo "列出 tmux session:"
echo "  tmux ls"
echo
echo "终止运行:"
echo "  tmux kill-session -t ${SESSION_NAME}"
echo
echo "示例覆盖参数:"
echo "  bash dyn-hamr/run_pose3d_hand_stages_tmux.sh"
echo "  STAGE=root EXTRA_ARGS='optim.root.num_iters=0' bash dyn-hamr/run_pose3d_hand_stages_tmux.sh"
echo "  DEVICE_OVERRIDE=cuda:1 bash dyn-hamr/run_pose3d_hand_stages_tmux.sh"
echo "  DEVICE_OVERRIDE=cpu STAGE=root EXTRA_ARGS='optim.root.num_iters=0' bash dyn-hamr/run_pose3d_hand_stages_tmux.sh"
