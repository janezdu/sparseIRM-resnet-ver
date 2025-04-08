# # dense/irmv1 run
# for rhotval in 60
# for rhotval in 20 40 60 80 100
# for rhotval in 5 10 15 25 30 35 45 50 55 65 70 75 85 90 95 105 135 155 175 195
# for rhotval in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 135 155 175 195
for rhotval in  54 58 59 60 61 62 66
do


echo $rhotval 
for i in $(seq 1 10);
do # gumbel sm with different rhotvals
    python main.py --config configs/smallscale/resnet18/resnet18-usc-unsigned.yaml --multigpu 0 --data dataset/ --epochs 50 --K 1 --conv_type DenseConv --weight_decay 0 --lr_policy cosine_lr --optimizer adam --lr 6e-3 --score_init_constant 1  --batch_size 390 --arch ResNet18 --set mnistcifar --iterative --TA --l2_regularizer_weight 0.001 --weight_opt adam --weight_opt_lr 0.0006 --hidden_dim 390 --penalty_anneal_iters 13 --penalty_weight 10000 --envs_num 2 --irm_type irmv1 --data_num 50000 --seed 0  --ts 0.28 --train_weights_at_the_same_time --cons_ratio 0.999_0.7_0.1 --noise_ratio 0.2 --use_dataloader 0 --prune_rate $rhotval --use_dataloader 0
done

done

