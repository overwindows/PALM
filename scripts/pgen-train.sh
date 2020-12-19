total_updates=20000
warmup_updates=500
lr=0.001
max_tokens=4096
update_freq=4
pointer_layer=-2

CUDA_VISIBLE_DEVICES=0 fairseq-train /datadrive/cnn_dm_bin \
    --user-dir ~/fairseq/examples/pointer_generator/pointer_generator_src \
    --max-tokens "$max_tokens" \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --required-batch-size-multiple 1 \
    --arch transformer_pointer_generator \
    --alignment-layer "$pointer_layer" \
    --alignment-heads 1 \
    --source-position-markers 1000 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler inverse_sqrt --lr "$lr" --max-update "$total_updates" --warmup-updates "$warmup_updates" \
    --update-freq "$update_freq" \
    --skip-invalid-size-inputs-valid-test --save-dir /bigdata/pgen_checkpoints --tensorboard-logdir /bigdata/logdir/pgen