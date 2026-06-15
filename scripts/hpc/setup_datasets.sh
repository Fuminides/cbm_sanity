#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$PWD/data/hpc_datasets}"
CONFIG_PATH="${CONFIG_PATH:-configs/hpc_datasets.env}"
DATASETS="${DATASETS:-cub awa2}"
CUB_URL="${CUB_URL:-https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1}"
AWA2_BASE_URL="${AWA2_BASE_URL:-https://cvml.ista.ac.at/AwA2/AwA2-base.zip}"
AWA2_DATA_URL="${AWA2_DATA_URL:-https://cvml.ista.ac.at/AwA2/AwA2-data.zip}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/hpc/setup_datasets.sh [options]

Options:
  --data-root PATH       Directory where datasets and archives are stored.
  --config-path PATH     Env file written with CUB_ROOT and AWA2_ROOT.
  --datasets LIST        Space-separated list: "cub", "awa2", or "cub awa2".
  -h, --help             Show this help.

Environment overrides:
  DATA_ROOT, CONFIG_PATH, DATASETS, CUB_URL, AWA2_BASE_URL, AWA2_DATA_URL

Outputs:
  CONFIG_PATH defaults to configs/hpc_datasets.env. The HPC qsub scripts source
  this file when it exists.
USAGE
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    printf 'Missing value for %s\n\n' "$option" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      require_value "$1" "${2:-}"
      DATA_ROOT="$2"
      shift 2
      ;;
    --config-path)
      require_value "$1" "${2:-}"
      CONFIG_PATH="$2"
      shift 2
      ;;
    --datasets)
      require_value "$1" "${2:-}"
      DATASETS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      printf 'Unexpected argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

log() {
  printf '[symbol_sanity] %s\n' "$*"
}

download_file() {
  local url="$1"
  local output_path="$2"
  if [[ -s "$output_path" ]]; then
    log "archive exists: $output_path"
    return
  fi
  log "downloading $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 10 -o "$output_path" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=5 --waitretry=10 -O "$output_path" "$url"
  else
    printf 'Neither curl nor wget is available\n' >&2
    exit 1
  fi
}

remove_invalid_tgz() {
  local archive_path="$1"
  if [[ -e "$archive_path" ]] && ! tar -tzf "$archive_path" >/dev/null 2>&1; then
    log "removing invalid tar archive: $archive_path"
    rm -f "$archive_path"
  fi
}

remove_invalid_zip() {
  local archive_path="$1"
  if [[ ! -e "$archive_path" ]]; then
    return
  fi
  if command -v unzip >/dev/null 2>&1; then
    if ! unzip -tq "$archive_path" >/dev/null 2>&1; then
      log "removing invalid zip archive: $archive_path"
      rm -f "$archive_path"
    fi
  elif ! python -m zipfile -t "$archive_path" >/dev/null 2>&1; then
    log "removing invalid zip archive: $archive_path"
    rm -f "$archive_path"
  fi
}

extract_tgz() {
  local archive_path="$1"
  local destination_dir="$2"
  local sentinel_path="$3"
  if [[ -e "$sentinel_path" ]]; then
    log "CUB already extracted: $sentinel_path"
    return
  fi
  log "extracting $archive_path"
  tar -xzf "$archive_path" -C "$destination_dir"
}

extract_zip() {
  local archive_path="$1"
  local destination_dir="$2"
  local sentinel_path="$3"
  if [[ -e "$sentinel_path" ]]; then
    log "AwA2 archive already extracted: $sentinel_path"
    return
  fi
  log "extracting $archive_path"
  if command -v unzip >/dev/null 2>&1; then
    unzip -oq "$archive_path" -d "$destination_dir"
  else
    python -m zipfile -e "$archive_path" "$destination_dir"
  fi
}

validate_cub() {
  local root="$1"
  local required=(
    "$root/images.txt"
    "$root/image_class_labels.txt"
    "$root/train_test_split.txt"
    "$root/classes.txt"
    "$root/attributes/image_attribute_labels.txt"
    "$root/images"
  )
  for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
      printf 'CUB validation failed; missing %s\n' "$path" >&2
      exit 1
    fi
  done
  if [[ ! -e "$root/attributes/attributes.txt" ]]; then
    printf 'CUB validation failed; missing %s\n' "$root/attributes/attributes.txt" >&2
    exit 1
  fi
}

normalize_cub_layout() {
  local root="$1"
  mkdir -p "$root/attributes"

  if [[ ! -e "$root/attributes/attributes.txt" && -e "$root/attributes.txt" ]]; then
    log "copying CUB root attributes.txt into attributes/attributes.txt"
    cp "$root/attributes.txt" "$root/attributes/attributes.txt"
  fi

  if [[ ! -e "$root/attributes/attributes.txt" ]]; then
    log "CUB attributes.txt is absent; generating generic attribute names"
    awk '{print $2}' "$root/attributes/image_attribute_labels.txt" \
      | sort -n \
      | uniq \
      | awk '{printf "%d attribute_%d\n", $1, $1}' \
      > "$root/attributes/attributes.txt"
  fi
}

validate_awa2() {
  local root="$1"
  local required=(
    "$root/JPEGImages"
  )
  for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
      printf 'AwA2 validation failed; missing %s\n' "$path" >&2
      exit 1
    fi
  done
  if [[ ! -e "$root/classes.txt" ]]; then
    printf 'AwA2 validation failed; missing %s\n' "$root/classes.txt" >&2
    exit 1
  fi
  if [[ ! -e "$root/predicates.txt" ]]; then
    printf 'AwA2 validation failed; missing %s\n' "$root/predicates.txt" >&2
    exit 1
  fi
  if [[ ! -e "$root/predicate-matrix-binary.txt" ]]; then
    printf 'AwA2 validation failed; missing %s\n' "$root/predicate-matrix-binary.txt" >&2
    exit 1
  fi
  if [[ ! -e "$root/predicate-matrix-continuous.txt" ]]; then
    printf 'AwA2 validation failed; missing %s\n' "$root/predicate-matrix-continuous.txt" >&2
    exit 1
  fi
}

normalize_awa2_layout() {
  local root="$1"

  if [[ ! -e "$root/JPEGImages" ]]; then
    local candidate
    candidate="$(find "$root" -maxdepth 4 -type d -name JPEGImages | head -n 1 || true)"
    if [[ -n "$candidate" ]]; then
      log "copying AwA2 JPEGImages from $candidate"
      cp -a "$candidate" "$root/JPEGImages"
    fi
  fi

  if [[ ! -e "$root/classes.txt" ]]; then
    local candidate
    candidate="$(find "$root" -maxdepth 3 -type f -name classes.txt | head -n 1 || true)"
    if [[ -n "$candidate" ]]; then
      log "copying AwA2 classes.txt from $candidate"
      cp "$candidate" "$root/classes.txt"
    fi
  fi

  if [[ ! -e "$root/classes.txt" ]]; then
    log "AwA2 classes.txt is absent; generating class names from JPEGImages/"
    find "$root/JPEGImages" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
      | sort \
      | awk '{printf "%d %s\n", NR, $0}' \
      > "$root/classes.txt"
  fi

  if [[ ! -e "$root/predicate-matrix-binary.txt" ]]; then
    local candidate
    candidate="$(find "$root" -maxdepth 3 -type f -name predicate-matrix-binary.txt | head -n 1 || true)"
    if [[ -n "$candidate" ]]; then
      log "copying AwA2 predicate-matrix-binary.txt from $candidate"
      cp "$candidate" "$root/predicate-matrix-binary.txt"
    fi
  fi

  if [[ ! -e "$root/predicate-matrix-continuous.txt" ]]; then
    local candidate
    candidate="$(find "$root" -maxdepth 3 -type f -name predicate-matrix-continuous.txt | head -n 1 || true)"
    if [[ -n "$candidate" ]]; then
      log "copying AwA2 predicate-matrix-continuous.txt from $candidate"
      cp "$candidate" "$root/predicate-matrix-continuous.txt"
    fi
  fi

  if [[ ! -e "$root/predicate-matrix-binary.txt" && -e "$root/predicate-matrix-continuous.txt" ]]; then
    log "AwA2 binary predicate matrix is absent; thresholding continuous matrix at 50"
    awk '{
      for (i = 1; i <= NF; i++) {
        printf "%s%s", ($i >= 50 ? 1 : 0), (i == NF ? ORS : OFS)
      }
    }' "$root/predicate-matrix-continuous.txt" > "$root/predicate-matrix-binary.txt"
  fi

  if [[ ! -e "$root/predicate-matrix-continuous.txt" && -e "$root/predicate-matrix-binary.txt" ]]; then
    log "AwA2 continuous predicate matrix is absent; copying binary matrix as continuous values"
    cp "$root/predicate-matrix-binary.txt" "$root/predicate-matrix-continuous.txt"
  fi

  if [[ ! -e "$root/predicates.txt" ]]; then
    local candidate
    candidate="$(find "$root" -maxdepth 3 -type f -name predicates.txt | head -n 1 || true)"
    if [[ -n "$candidate" ]]; then
      log "copying AwA2 predicates.txt from $candidate"
      cp "$candidate" "$root/predicates.txt"
    fi
  fi

  if [[ ! -e "$root/predicates.txt" ]]; then
    log "AwA2 predicates.txt is absent; generating predicate names from predicate matrix"
    awk 'NR == 1 {for (i = 1; i <= NF; i++) printf "%d predicate_%d\n", i, i}' \
      "$root/predicate-matrix-binary.txt" \
      > "$root/predicates.txt"
  fi
}

setup_cub() {
  local downloads_dir="$DATA_ROOT/downloads"
  local archive_path="$downloads_dir/CUB_200_2011.tgz"
  local cub_root="$DATA_ROOT/CUB_200_2011"
  mkdir -p "$downloads_dir"
  remove_invalid_tgz "$archive_path"
  download_file "$CUB_URL" "$archive_path"
  extract_tgz "$archive_path" "$DATA_ROOT" "$cub_root/images.txt"
  normalize_cub_layout "$cub_root"
  validate_cub "$cub_root"
  CUB_ROOT_RESOLVED="$cub_root"
}

setup_awa2() {
  local downloads_dir="$DATA_ROOT/downloads"
  local base_archive="$downloads_dir/AwA2-base.zip"
  local data_archive="$downloads_dir/AwA2-data.zip"
  local awa2_root="$DATA_ROOT/Animals_with_Attributes2"
  mkdir -p "$downloads_dir"
  remove_invalid_zip "$base_archive"
  remove_invalid_zip "$data_archive"
  download_file "$AWA2_BASE_URL" "$base_archive"
  download_file "$AWA2_DATA_URL" "$data_archive"
  mkdir -p "$awa2_root"
  extract_zip "$base_archive" "$awa2_root" "$awa2_root/classes.txt"
  extract_zip "$data_archive" "$awa2_root" "$awa2_root/JPEGImages"
  normalize_awa2_layout "$awa2_root"
  validate_awa2 "$awa2_root"
  AWA2_ROOT_RESOLVED="$awa2_root"
}

write_config() {
  mkdir -p "$(dirname "$CONFIG_PATH")"
  {
    printf '# Generated by scripts/hpc/setup_datasets.sh\n'
    printf 'CUB_ROOT=%q\n' "${CUB_ROOT_RESOLVED:-$DATA_ROOT/CUB_200_2011}"
    printf 'AWA2_ROOT=%q\n' "${AWA2_ROOT_RESOLVED:-$DATA_ROOT/Animals_with_Attributes2}"
  } > "$CONFIG_PATH"
  log "wrote dataset env file: $CONFIG_PATH"
}

mkdir -p "$DATA_ROOT"

for dataset in $DATASETS; do
  case "$dataset" in
    cub)
      setup_cub
      ;;
    awa2)
      setup_awa2
      ;;
    *)
      printf 'Unknown dataset %s; expected cub or awa2\n' "$dataset" >&2
      exit 2
      ;;
  esac
done

write_config
log "dataset setup complete"
