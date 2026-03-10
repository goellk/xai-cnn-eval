#!/usr/bin/env bash
set -euo pipefail

######################################################################
# CONFIGURATION – change these paths/flags if your layout is different
######################################################################
SRC_DIR="IMAGENET1K"                        # Original root (contains “training/” and “validation/”)
CLASS_FILE="imagenet80_subsets_classes.txt" # File listing subsets and class IDs
DST_PREFIX="imagenet80_"                    # New subset roots will be imagenet80_0 … imagenet80_9
DO_MOVE=0                                   # 1 = move folders (destructive) ─ 0 = copy (non-destructive)
######################################################################

# 1. Parse subsets and class IDs from CLASS_FILE
current_subset=""
declare -A subsets

while IFS= read -r line; do
    if [[ $line =~ ^imagenet80_[0-9]+: ]]; then
        current_subset="${line%:}"      # e.g., "imagenet80_0"
        subsets["$current_subset"]=""   # Initialize empty list
    elif [[ -n "$line" ]]; then
        subsets["$current_subset"]+="$line "
    fi
done < "$CLASS_FILE"

# 2. Loop through each subset and copy/move classes
for subset_name in "${!subsets[@]}"; do
    echo "Processing $subset_name..."
    subset_root="$subset_name"
    mkdir -p "${subset_root}/training" "${subset_root}/validation"

    for class_id in ${subsets[$subset_name]}; do
        echo "  → Handling class $class_id"
        if [[ $DO_MOVE -eq 1 ]]; then
            mv "${SRC_DIR}/training/${class_id}" "${subset_root}/training/"
            mv "${SRC_DIR}/validation/${class_id}" "${subset_root}/validation/"
        else
            cp -r "${SRC_DIR}/training/${class_id}" "${subset_root}/training/"
            cp -r "${SRC_DIR}/validation/${class_id}" "${subset_root}/validation/"
        fi
    done
done

echo "ImageNet split completed using class file: $CLASS_FILE"

