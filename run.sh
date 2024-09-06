
python3 main.py --config configs/smallscale/resnet18/resnet18-usc-unsigned.yaml --multigpu 0 --data dataset/ --epochs 50 --K 1 --conv-type ProbMaskConv --weight-decay 0 --lr-policy cosine_lr --optimizer adam --lr 6e-3 --score-init-constant 1 --prune-rate 1 --batch-size 390 --arch ResNet18 --set mnistcifar --iterative --TA --l2_regularizer_weight 0.001 --weight_opt adam --weight_opt_lr 0.0006 --hidden_dim 390 --penalty_anneal_iters 400 --penalty_weight 10000 --envs_num 2 --irm_type irmv1 --data_num 50000 --seed 0  --ts 0.28 --train_weights_at_the_same_time --cons_ratio 0.999_0.7_0.1 --noise_ratio 0.2

# for i in $(seq 1 10);
# do
#     python3 main.py --config configs/smallscale/resnet18/resnet18-usc-unsigned.yaml --multigpu 0 --data dataset/ --epochs 50 --K 1 --conv-type ProbMaskConv --weight-decay 0 --lr-policy cosine_lr --optimizer adam --lr 6e-3 --score-init-constant 1 --prune-rate 0.95 --batch-size 390 --arch ResNet18 --set mnistcifar --iterative --TA --l2_regularizer_weight 0.001 --weight_opt adam --weight_opt_lr 0.0006 --hidden_dim 390 --penalty_anneal_iters 13 --penalty_weight 10000 --envs_num 2 --irm_type irmv1 --data_num 50000 --seed 0  --ts 0.28 --train_weights_at_the_same_time --cons_ratio 0.999_0.7_0.1 --noise_ratio 0.2
# done



# --ours  