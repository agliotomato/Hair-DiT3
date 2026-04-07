#!/bin/bash
set -e

echo "========================================================="
echo "  Stage 1: Generalization Training on 'unbraid' dataset"
echo "========================================================="
python scripts/train.py \
    --train_data_dir "dataset/unbraid" \
    --batch_size 4 \
    --total_steps 15000 \
    --phase1_steps 5000 \
    --phase2_steps 15000 \
    --mixed_precision "bf16" \
    --output_dir "outputs/stage1_unbraid"

echo ""
echo "========================================================="
echo "  Stage 2: Specialization Training on 'braid' dataset"
echo "========================================================="
# Stage 2 uses Phase 2 settings from the very beginning (no frozen blocks)
# We set phase1 and phase2 constraints to 0 to bypass them.
python scripts/train.py \
    --train_data_dir "dataset/braid" \
    --batch_size 4 \
    --total_steps 10000 \
    --phase1_steps 0 \
    --phase2_steps 0 \
    --resume_from_checkpoint "outputs/stage1_unbraid/final/hair_s2i_modules.pt" \
    --mixed_precision "bf16" \
    --output_dir "outputs/stage2_braid"

echo "========================================================="
echo "  Curriculum Training Completed!"
echo "========================================================="
