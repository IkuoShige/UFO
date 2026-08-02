#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO="${K1_LAFAN1_DATASET_REPO:-wu0712/retargeted_lafan1_for_booster_k1}"
DATASET_FILE="${K1_LAFAN1_DATASET_FILE:-Lafan1_Retarget_Dataset.zip}"
DEST_DIR="${K1_LAFAN1_DEST_DIR:-humanoidverse/data/k1_lafan1}"

usage() {
  cat <<USAGE
Usage: bash scripts/download_k1_lafan1_data.sh

Downloads the retargeted LAFAN1 -> Booster K1 motion CSVs from Hugging Face:
  https://huggingface.co/datasets/${DATASET_REPO}

and stages the per-clip CSV files under:
  ${DEST_DIR}/

Environment overrides:
  K1_LAFAN1_DATASET_REPO=${DATASET_REPO}
  K1_LAFAN1_DATASET_FILE=${DATASET_FILE}
  K1_LAFAN1_DEST_DIR=${DEST_DIR}
USAGE
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

if [[ -d "${DEST_DIR}" ]] && [[ -n "$(find "${DEST_DIR}" -maxdepth 1 -name '*.csv' -print -quit)" ]]; then
  echo "K1 LAFAN1 motion CSVs already present in ${DEST_DIR}, skipping download."
  exit 0
fi

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT

zip_url="https://huggingface.co/datasets/${DATASET_REPO}/resolve/main/${DATASET_FILE}"
echo "Downloading ${zip_url}"
curl -L --fail -o "${tmpdir}/${DATASET_FILE}" "${zip_url}"

echo "Extracting ${DATASET_FILE}"
unzip -q "${tmpdir}/${DATASET_FILE}" -d "${tmpdir}/extracted"

csv_dir="$(find "${tmpdir}/extracted" -mindepth 1 -maxdepth 3 -type d -name 'k1_lafan1' -print -quit)"
if [[ -z "${csv_dir}" ]]; then
  csv_dir="$(dirname "$(find "${tmpdir}/extracted" -name '*.csv' -print -quit)")"
fi
if [[ -z "${csv_dir}" || ! -d "${csv_dir}" ]]; then
  echo "Could not locate extracted CSV directory in ${DATASET_FILE}" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"
cp "${csv_dir}"/*.csv "${DEST_DIR}/"

count="$(find "${DEST_DIR}" -maxdepth 1 -name '*.csv' | wc -l)"
echo "Staged ${count} K1 LAFAN1 motion CSVs into ${DEST_DIR}"
