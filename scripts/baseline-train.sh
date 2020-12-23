DATA_BIN=/datadrive/cnn_dm_bin
EXP_NAME=debug
total_updates=200000
warmup_updates=500
lr=0.001
max_tokens=4096
update_freq=4

CUDA_VISIBLE_DEVICES=3 fairseq-train /datadrive/cnn_dm_bin \
    --arch transformer --share-decoder-input-output-embed \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr "$lr" --lr-scheduler polynomial_decay --warmup-updates 400 --max-update $total_updates --total-num-update 20000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 2048 --skip-invalid-size-inputs-valid-test \
    --update-freq 4 \
    --find-unused-parameters \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-interval-updates -1 \
    --keep-last-epochs -1 \
    --keep-best-checkpoints -1 \
    --no-epoch-checkpoints \
    --save-dir /bigdata/baseline_checkpoints --tensorboard-logdir /bigdata/logdir/baseline

# --eval-bleu-print-samples \
