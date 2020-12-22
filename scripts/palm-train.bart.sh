total_updates=200000
warmup_updates=500      
lr=3e-05
max_tokens=4096
update_freq=4
pointer_layer=-2
bart_path=/bigdata/bart.base/model.pt
exp_name=palm_bart

CUDA_VISIBLE_DEVICES=1 python -m utils.train /datadrive/cnn_dm_bin \
    --user-dir src \
    --restore-file $bart_path \
    --max-tokens $max_tokens \
    --task auto_encoding_regressive \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch palm_base \
    --criterion label_smoothed_cross_entropy_with_masked_lm --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --eval-bleu \
    --lr-scheduler inverse_sqrt --lr $lr --max-update $total_updates --warmup-updates $warmup_updates \
    --update-freq $update_freq \
    --skip-invalid-size-inputs-valid-test \
    --alignment-layer "$pointer_layer" \
    --alignment-heads 1 \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --find-unused-parameters \
    --keep-interval-updates -1 \
    --keep-last-epochs -1 \
    --keep-best-checkpoints -1 \
    --no-epoch-checkpoints \
    --best-checkpoint-metric loss \
    --act-dropout 0.1 --save-dir /bigdata/"$exp_name"_checkpoints --tensorboard-logdir /bigdata/logdir/"$exp_name"

    # --source-position-markers 1000 \
    # --train-subset train \
    # --valid-subset valid \
    # --validate-interval 1 \
    # --max-tokens-valid 2048 \
    # --fp16 \
    # --encoder-embed-dim 512 \
    # --encoder-ffn-embed-dim 2048 \
    # --encoder-attention-heads 8 \
    # --decoder-embed-dim 512 \
    # --decoder-ffn-embed-dim 2048 \
    # --decoder-attention-heads 8 \
    # --restore-file $roberta_path \



