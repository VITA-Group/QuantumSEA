python -u train.py \
--dataset mnist \
--num_class 2 \
--n_test_images 300 \
--n_train_images 500 \
--n_val_images 300 \
--arch RXYZ_Model_DST \
--n_wires 4 \
--n_blocks 8 \
--save_dir MNIST2_SEA \
--epochs 50 \
--batch_size 500 \
--lr 0.3 \
--weight_decay 1e-4 \
--min_lr 0.03 \
--warmup_epochs 0 \
--qc_device ibmq_quito \
--n_shots 8192 \
--death_mode saliency \
--growth_mode gradient_prob \
--density 0.5 \
--death_rate 0.25 \
--grad_beta 0.9 \
--frequency 2 \
--seed 37