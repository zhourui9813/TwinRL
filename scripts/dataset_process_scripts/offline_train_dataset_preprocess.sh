#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./offline_train_dataset_preprocess.sh <INPUT_PKL>                 # Execute normalization by default
#   ./offline_train_dataset_preprocess.sh <INPUT_PKL> --no_normalize  # Skip normalization

DO_NORMALIZE=1
INPUT_PKL=""
for arg in "$@"; do
  case "$arg" in
    --no_normalize) DO_NORMALIZE=0 ;;
    --*)
      echo "Unknown arg: $arg" >&2
      exit 1
      ;;
    *)
      if [[ -z "$INPUT_PKL" ]]; then
        INPUT_PKL="$arg"
      else
        echo "Unexpected extra positional arg: $arg" >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$INPUT_PKL" ]]; then
  echo "Usage: $0 <INPUT_PKL> [--no_normalize]" >&2
  exit 1
fi

DELTA_PKL="${INPUT_PKL%.pkl}_delta.pkl"

NORMALIZED_PKL="${DELTA_PKL%.pkl}_normalized.pkl"

echo "INPUT_PKL:       $INPUT_PKL"
echo "DELTA_PKL:       $DELTA_PKL"
echo "NORMALIZED_PKL:  $NORMALIZED_PKL"
echo "DO_NORMALIZE:    $DO_NORMALIZE"

echo "#########################################"
echo "Change to Delta pose"
echo "#########################################"

python convert_absolute_to_delta.py \
  --input_path "$INPUT_PKL" \
  --output_path "$DELTA_PKL"

# Select input for zero_state: default to delta; use normalized if normalization is applied
ZERO_INPUT_PKL="$DELTA_PKL"

if [[ "$DO_NORMALIZE" -eq 1 ]]; then
  echo "#########################################"
  echo "Normalization"
  echo "#########################################"

  python normalize_actions.py \
    --input_path "$DELTA_PKL" \
    --output_normalized_path "$NORMALIZED_PKL"
    
  ZERO_INPUT_PKL="$NORMALIZED_PKL"
else
  echo "#########################################"
  echo "Skip Normalization"
  echo "#########################################"
fi

echo "#########################################"
echo "Change to Zero State"
echo "#########################################"

python state_process.py \
  --input_path "$ZERO_INPUT_PKL"
# --use_state
