PORT=10000

cd /research/cbim/vast/yg397/DePT/openDePT
for SEED in 2020
do
    for STAGE in 4
    do
        for PN in 50
        do
            for LR in 5e-4
            do
                for BETA in 0.05
                do
                    MEMO="div0.005_dinoCLS+CosPrompt_stage${STAGE}_PN${PN}_ttemp0.07_beta0.1_beta2_0.1_sgd_lr${LR}"
                    #MEMO='test'
                    echo $MEMO

                    let WARM="$STAGE*$PN"
                    echo $WARM

                    python main.py train_source=true learn=source \
                    seed=${SEED} port=${PORT} memo=${MEMO} project="VISDA-C_source" \
                    data.data_root="${PWD}/datasets" data.workers=8 \
                    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
                    optim='sgd' source_algorithm='dino' model='dino' \
                    learn.epochs=10 \
                    model.arch="vit_base_patch16_224" \
                    model.tune_type='full' \
                    model.m=0.99 \
                    model.stage_num=${STAGE} \
                    model.prompt_num=${PN} \
                    model.init='xavier_uniform' \
                    model.consistency_type='cls_prompt' \
                    model.hierarchy=True \
                    model.nlayer_head=2 \
                    model.teacher_temp=0.07 \
                    model.out_dim=2048 \
                    optim.lr=${LR} \
                    data.batch_size=128 \
                    multiprocessing_distributed=False \
                    optim.lr_decay='cos' \
                    learn.warmup_iters=${WARM} \
                    learn.beta=0.1 \
                    learn.beta2=0.1 \
                    learn.lam=0.005 \
                    learn.prompt_div=True \
                    learn.teacher_init_temp=0.07 \
                    learn.warmup_epoch=0 \
                    learn.tau=1
                done
            done
        done
    done
done
