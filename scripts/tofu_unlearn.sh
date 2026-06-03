#!/bin/bash
#SBATCH --job-name=tofu_unlearn
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/u/vz8/unlearning/logs/tofu_%j.log
#SBATCH --error=/u/vz8/unlearning/logs/tofu_%j.err

# variables--EDIT HERE TO CHANGE EXPERIMENT
MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
TRAINER="GradAscent"
TASK_NAME="tofu_gradascent"
EPOCHS=5
BATCH_SIZE=4
GRAD_ACCUM=4

TARGET_MODEL="open-unlearning/tofu_${MODEL}_full"
RETAIN_LOGS="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
EVAL_TASK="${TASK_NAME}_eval"

# environment
export HF_HOME=/scratch/vz8/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
conda activate tofu_env
cd /u/vz8/unlearning/open-unlearning
mkdir -p /u/vz8/unlearning/logs

echo "=============================="
echo "Running experiment:"
echo "  Model:        $MODEL"
echo "  Forget split: $FORGET_SPLIT"
echo "  Retain split: $RETAIN_SPLIT"
echo "  Trainer:      $TRAINER"
echo "  Task name:    $TASK_NAME"
echo "  Target model: $TARGET_MODEL"
echo "=============================="

# ── Step 1: Unlearn ───────────────────────────────────────────
echo "Starting unlearning..."
python src/train.py --config-name=unlearn.yaml \
  model.attn_implementation="sdpa' \
  experiment=unlearn/tofu/default \
  forget_split=${FORGET_SPLIT} \
  retain_split=${RETAIN_SPLIT} \
  trainer=${TRAINER} \
  model=${MODEL} \
  model.model_args.pretrained_model_name_or_path=${TARGET_MODEL} \
  ++model.model_args.attn_implementation=eager \
  task_name=${TASK_NAME} \
  trainer.args.num_train_epochs=${EPOCHS} \
  trainer.args.per_device_train_batch_size=${BATCH_SIZE} \
  trainer.args.gradient_accumulation_steps=${GRAD_ACCUM} \
  trainer.args.output_dir=saves/models/${TASK_NAME}

echo "Unlearning done!"

# ── Step 2: Evaluate ──────────────────────────────────────────
echo "Starting evaluation..."
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=${MODEL} \
  model.model_args.pretrained_model_name_or_path=saves/models/${TASK_NAME} \
  retain_logs_path=${RETAIN_LOGS} \
  task_name=${EVAL_TASK}

echo "=============================="
echo "Done! Results saved to saves/eval/${EVAL_TASK}"
echo "=============================="
