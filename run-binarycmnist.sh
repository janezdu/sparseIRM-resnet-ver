# # dense/irmv1 run
for rhotval in 60
# for rhotval in 20 40 60 80 100
# for rhotval in 5 10 15 25 30 35 45 50 55 65 70 75 85 90 95 105 135 155 175 195
do

for i in $(seq 1 10);
do
    python main.py --config configs/smallscale/resnet18/resnet18-usc-unsigned.yaml --multigpu 1 --data dataset/ --epochs 1500 --K 1 --conv-type ProbMaskConvLinear --weight-decay 0 --lr-policy cosine_lr --optimizer adam --lr 6e-3 --score-init-constant 1 --prune-rate 0.95 --batch-size 50000 --arch MLP --set mnist --iterative --TA --l2_regularizer_weight 0.001 --weight_opt adam --weight_opt_lr 0.0006 --hidden_dim 390 --penalty_anneal_iters 200 --penalty_weight 10000 --envs_num 2 --irm_type rex --data_num 50000 --seed 0  --ts 0.28 --train_weights_at_the_same_time 
done

done

