#!/bin/bash
set -e

echo "========================================================="
echo "  Stage 1: Generalization Training (Phase 1)"
echo "========================================================="
python scripts/train.py --config configs/phase1.yaml

echo ""
echo "========================================================="
echo "  Stage 2: Specialization Training (Phase 2)"
echo "========================================================="
python scripts/train.py --config configs/phase2.yaml

echo "========================================================="
echo "  Curriculum Training Completed!"
echo "========================================================="
