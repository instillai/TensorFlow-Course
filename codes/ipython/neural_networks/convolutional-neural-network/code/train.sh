
# Run training.
python train_classifier.py \
  --batch_size=512 \
  --max_num_checkpoint=10 \
  --num_classes=10 \
  --num_epochs=1 \
  --initial_learning_rate=0.001 \
  --num_epochs_per_decay=1 \
  --is_training=True \
  --allow_soft_placement=True \
  --fine_tuning=False \
  --online_test=True \
  --log_device_placement=False

