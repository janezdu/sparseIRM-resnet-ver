for rhotval in 25;
#pgd-ERM
for i in $(seq 1 10);
do
    python main.py --config configs/smallscale/resnet18/resnet18-usc-unsigned.yaml --multigpu 0 --data dataset/ --epochs 1500 --K 1 --cons_ratio 0.999_0.7_0.1 --conv_type DenseLinear --weight_decay 0 --lr_policy cosine_lr --optimizer adam --lr 6e-3 --score_init_constant 1 --prune_rate 0.95 --batch_size 50000 --arch MLPFull --set mnistfull --iterative --TA --l2_regularizer_weight 0.001 --weight_opt adam --weight_opt_lr 0.0006 --hidden_dim 390 --penalty_anneal_iters 200 --penalty_weight 0 --envs_num 2 --irm_type irmv1 --data_num 50000 --seed 0  --ts 0.28 --train_weights_at_the_same_time --use_pgd --last_layer_dense --num_classes 1 --pgd_anneal_iters 1200 --bn_type LearnedBatchNorm --pgd_skip_steps 5 --fraction_z 1.0 --rho_tolerance $rhotval --noise_ratio 0.2
done

done