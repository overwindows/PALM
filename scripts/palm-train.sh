TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4

CUDA_VISIBLE_DEVICES=2 fairseq-train /datadrive/cnn_dm_bin \
    --user-dir src \
    --max-tokens $MAX_TOKENS \
    --task auto_encoding_regressive \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch palm_base \
    --criterion label_smoothed_cross_entropy_with_masked_lm \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters --act-dropout 0.1 --save-dir /bigdata/palm_checkpoints


    # --restore-file $BART_PATH \
