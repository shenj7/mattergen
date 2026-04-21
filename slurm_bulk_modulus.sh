#!/bin/bash
#SBATCH --job-name=bulk_modulus
#SBATCH --output=/scratch/gilbreth/shen574/logs/bulk_modulus_%A_%a.out
#SBATCH --error=/scratch/gilbreth/shen574/logs/bulk_modulus_%A_%a.err
#SBATCH --array=0-37
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gilbreth
#SBATCH --account=shen574

# ── Environment ──────────────────────────────────────────────────────────────
cd /home/user/mattergen  # adjust if your project root differs
source .venv/bin/activate
export PYTHONPATH="/home/user/mattergen:${PYTHONPATH}"

mkdir -p /scratch/gilbreth/shen574/logs

# ── Checkpoint array (must match the generate/evaluate script exactly) ────────
CHECKPOINTS=(
    /scratch/gilbreth/shen574/ddpo_mattersim_3_27_relaxed/checkpoint_epoch_{10..130..10}.pt
    /scratch/gilbreth/shen574/ddpo_mattersim_3_27_relaxed_continue/checkpoint_epoch_{10..150..10}.pt
    /scratch/gilbreth/shen574/ddpo_mattersim_3_27_relaxed_continue_2/checkpoint_epoch_{10..100..10}.pt
)

# ── Resolve output directory for this array task ──────────────────────────────
CPKT_PATH=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}

FOLDER_NAME=$(basename $(dirname "$CPKT_PATH"))
FILE_BASE=$(basename "$CPKT_PATH" .pt)

SCRATCH_BASE="/scratch/gilbreth/shen574"
OUTPUT_DIR="${SCRATCH_BASE}/outputs/${FOLDER_NAME}/${FILE_BASE}"

echo "Task ID:       $SLURM_ARRAY_TASK_ID"
echo "Checkpoint:    $CPKT_PATH"
echo "Output dir:    $OUTPUT_DIR"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory does not exist: $OUTPUT_DIR"
    exit 1
fi

if [ ! -f "${OUTPUT_DIR}/generated_crystals_cif.zip" ]; then
    echo "ERROR: generated_crystals_cif.zip not found in $OUTPUT_DIR"
    exit 1
fi

# Skip if results already exist (allows safe re-submission of failed tasks)
if [ -f "${OUTPUT_DIR}/bulk_modulus_results.csv" ]; then
    echo "bulk_modulus_results.csv already exists, skipping."
    exit 0
fi

# ── Run bulk modulus calculation ───────────────────────────────────────────────
python process_bulk_modulus_outputs.py "$OUTPUT_DIR"
