#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") --dataset_path PATH [OPTIONS]" >&2
  echo "Options:" >&2
  echo "  --skip_scans                skip downloading 3RScan data (sequences are required)" >&2
  echo "  --download_script PATH     path to download script (default: dataset_path/download.py)" >&2
  echo "  --workers N                num_workers for frame processing (default: 1)" >&2
  echo "  --scan_id UUID             download and process only a specific scan" >&2
  echo "  --skip_json            skip generating scenegraph.json and splits from scratch" >&2
  echo "  --skip_embeddings      skip generating numberbatch and clip embeddings from scratch" >&2
  echo "  --numberbatch_path PATH    path to numberbatch file or download location (default: /tmp)" >&2
  exit 1
}

# Parse flags
DATASET_PATH=""
SKIP_SCANS="false"
DOWNLOAD_SCRIPT=""
WORKERS="1"
SCAN_ID=""
SKIP_JSON="false"
SKIP_EMBEDDINGS="false"
NUMBERBATCH_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_path)
      [[ $# -ge 2 ]] || usage
      DATASET_PATH="$2"; shift 2 ;;
    --skip_scans)
      SKIP_SCANS="true"; shift ;;
    --download_script)
      [[ $# -ge 2 ]] || usage
      DOWNLOAD_SCRIPT="$2"; shift 2 ;;
    --workers)
      [[ $# -ge 2 ]] || usage
      WORKERS="$2"; shift 2 ;;
    --scan_id)
      [[ $# -ge 2 ]] || usage
      SCAN_ID="$2"; shift 2 ;;
    --skip_json)
      SKIP_JSON="true"; shift ;;
    --skip_embeddings)
      SKIP_EMBEDDINGS="true"; shift ;;
    --numberbatch_path)
      [[ $# -ge 2 ]] || usage
      NUMBERBATCH_PATH="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage ;;
  esac
done

[[ -n "${DATASET_PATH}" ]] || usage

# Prompt for remaining inputs (none)

# Normalize dataset path variants
DATASET_PATH_NOSLASH="${DATASET_PATH%/}"
DATASET_PATH_SLASH="${DATASET_PATH_NOSLASH}/"

# Defaults for optional paths
if [[ -z "${DOWNLOAD_SCRIPT}" ]]; then
  DOWNLOAD_SCRIPT="${DATASET_PATH_NOSLASH}/download.py"
fi
if [[ -z "${NUMBERBATCH_PATH}" ]]; then
  NUMBERBATCH_PATH="/tmp"
fi

# Download step (only if scans are not skipped)
if [[ "${SKIP_SCANS}" == "false" ]]; then
  echo "Running download_3rscan.py ..."
  SCAN_ID_FLAG=""
  if [[ -n "${SCAN_ID}" ]]; then
    SCAN_ID_FLAG=(--id "${SCAN_ID}")
  fi
  python tools/download_3rscan.py \
    --dataset_path "${DATASET_PATH_SLASH}" \
    --download_script "${DOWNLOAD_SCRIPT}" \
    ${SCAN_ID_FLAG[@]:-} \
    --sequences

  echo "Running download_3dssg.py ..."
  python tools/download_3dssg.py \
    --dataset_path "${DATASET_PATH_SLASH}"
fi

# Generate JSON step (only if --generate_json is set)
if [[ "${SKIP_JSON}" == "false" ]]; then
  echo "Running generate_gt_scenegraphs.py ..."
  python tools/generate_gt_scenegraphs.py \
    --dataset_path "${DATASET_PATH_NOSLASH}" \
    --overwrite

  echo "Running generate_splits.py ..."
  python tools/generate_splits.py \
    --dataset_path "${DATASET_PATH_NOSLASH}"
fi

# Always run these steps
echo "Running generate_rendered_views.py ..."
python tools/generate_rendered_views.py \
  --dataset_path "${DATASET_PATH_NOSLASH}" \
  --workers "${WORKERS}"

echo "Running generate_hetero_graphs.py ..."
python tools/generate_hetero_graphs.py \
  --dataset_path "${DATASET_PATH_NOSLASH}" \
  --num_workers "${WORKERS}"

# Generate embeddings step (only if --generate_embeddings is set)
if [[ "${SKIP_EMBEDDINGS}" == "false" ]]; then
  echo "Running generate_embeddings.py ..."
  python tools/generate_embeddings.py \
    --dataset_path "${DATASET_PATH_SLASH}" \
    --numberbatch_path "${NUMBERBATCH_PATH}"
fi

echo "All steps completed successfully."


