# Use DINO prompt trained source domain model for test-time adaptation
PORT=10002

#dinoPrompt_stage4_PN10_ttemp0.04_beta2_0.1_normFalse_sgd_lr5e-4/
#dinoCLS_stage12_PN10_beta0.1_ttemp0.04_sgd_lr5e-4_warmup50_normlastFalse
cd /research/cbim/vast/yg397/DePT/openDePT
for SEED in 2020
do
    for LR in 5e-3
    do
        for TEACHER_TEMP in 0.07
        do
            for PERCENT in 1.0
            do
                for M in 0.99
                do
                    #MEMO="div0.0_COSCLS+Prompt_prompt_nh2_dim2048_stage4_PN10_sgd_beta0.0_beta2_0.0_lr${LR}_percent${PERCENT}"
                    MEMO='test'
                    echo ${MEMO}

                    EPOCH=$(python -c "print(int(15/$PERCENT))")
                    echo $EPOCH
                    EVAL=$(python -c "print(int(1/$PERCENT))")
                    echo $EVAL

                    python main.py \
                    seed=${SEED} port=${PORT} memo=${MEMO} project="VISDA-C_tta" \
                    optim='sgd' target_algorithm='dino' model='dino' \
                    data.data_root="${PWD}/datasets" data.workers=8 \
                    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
                    model.arch="vit_base_patch16_224" \
                    model.tune_type='prompt' \
                    model.m=${M} \
                    model.stage_num=4 \
                    model.prompt_num=50 \
                    model.consistency_type='cls_prompt' \
                    model.hierarchy=True \
                    model.nlayer_head=2 \
                    model.src_log_dir="/research/cbim/vast/yg397/DePT/openDePT/exp/VISDA-C_source/div0.005_dinoCLS+CosPrompt_stage4_PN50_ttemp0.07_beta0.1_beta2_0.1_sgd_lr5e-4/best_train_2020.pth.tar" \
                    multiprocessing_distributed=True \
                    data.batch_size=128 \
                    optim.lr=${LR} \
                    learn.epochs=${EPOCH} \
                    learn.eval_freq=${EVAL} \
                    learn.queue_size=-1 \
                    learn.beta=0.1 \
                    learn.beta2=0.1 \
                    model.teacher_temp=0.07 \
                    model.out_dim=2048 \
                    data.percent=${PERCENT} \
                    data.random_seed=0 \
                    learn.prompt_div=True \
                    learn.lam=0.05 \
                    learn.eta=1

                done
            done
        done
    done
done
