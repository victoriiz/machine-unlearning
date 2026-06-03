# After unlearning, fine-tune briefly on the forget set
# then re-evaluate — does the model quickly recover forgotten info?
#relearn_cmd = f"""
python src/train.py --config-name=finetune.yaml \
  experiment=finetune/tofu/default \
  split={FORGET_SPLIT} \
  model={MODEL} \
  model.model_args.pretrained_model_name_or_path=saves/models/{TASK_NAME} \
  task_name={TASK_NAME}_relearn \
  trainer.args.num_train_epochs=1 \
  trainer.args.per_device_train_batch_size={BATCH_SIZE} \
  trainer.args.gradient_accumulation_steps={GRAD_ACCUM} \
  trainer.args.output_dir=saves/models/{TASK_NAME}_relearn
#"""
