TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4
BART_PATH=/apdcephfs/private_kevinkyhong/prev_trained_model/bart-base/

CUDA_VISIBLE_DEVICES=1 fairseq-train /apdcephfs/private_kevinkyhong/data/cmrc2018_public/corpus_bin \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_base \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters --save-dir /apdcephfs/share_1351585/kevinkyhong/fine_tuned_model/bart_finetuned_cmrc_checkpoints --tensorboard-logdir /apdcephfs/share_1351585/kevinkyhong/logdir/bart \
    --restore-file $BART_PATH \
