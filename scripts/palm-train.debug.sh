DATA_BIN=/datadrive/cnn_dm_bin

CUDA_VISIBLE_DEVICES=2 python -m utils.train $DATA_BIN \
--user-dir src \
--task auto_encoding_regressive --arch palm_base --criterion label_smoothed_cross_entropy_with_masked_lm \
--share-all-embeddings --share-decoder-input-output-embed --layernorm-embedding \
--optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
--lr 3e-05 --lr-scheduler polynomial_decay --total-num-update 10000 --warmup-updates 1500 \
--skip-invalid-size-inputs-valid-test \
--max-tokens 2048 \
--train-subset train \
--valid-subset valid \
--validate-interval 1 \
--max-tokens-valid 2048 \
--clip-norm 0.1 \
--save-dir /bigdata/debug_checkpoints --tensorboard-logdir /bigdata/logdir/debug \
--keep-interval-updates -1 \
--keep-last-epochs -1 \
--keep-best-checkpoints -1 \
--no-epoch-checkpoints \
--best-checkpoint-metric loss \
--weight-decay 0.01 \
--dropout 0.1 \
--attention-dropout 0.1 \
--act-dropout 0.1 --reset-optimizer

# --sample-break-mode complete_doc \
# --mask 0.3 \
# --mask-random 0.1 \
# --poisson-lambda 3.5 \
# --permute-sentences 1 \
# --mask-length span-poisson \
# --replace-length 1 \
# --cpu \
# --fp16 \
# --max-update 100 \
# --restore-file checkpoint_last.pt \
# --save-interval 1 --save-interval-updates 50 \
# --encoder-embed-dim 768 \
# --encoder-ffn-embed-dim 3072 \
# --encoder-layers 6 \
# --encoder-attention-heads 12 \
# --decoder-layers 6 --decoder-attention-heads 12 \
# --encoder-learned-pos \
# --decoder-embed-dim 768 \
# --decoder-ffn-embed-dim 3072 \
# --decoder-learned-pos \
# --decoder-output-dim 768 \
# --no-scale-embedding \
# --activation-fn gelu \
# --copy-attention \
# --copy-attention-heads 1 \
# --copy-attention-dropout 0.2 \
# --tokens-per-sample 512 \
# --log-interval 100 \
# --log-format json \
# --max-source-positions 1024 \
# --max-target-positions 1024 \
# --num-workers 4 \
# --model-parallel-size 1 \
# --update-freq 4 \
# --num-segment 1 \
# --min-lr -1 \
# --optimizer-overrides {} \
# --min-loss-scale 0.0001 \
# --power 1 \
# --bucket-cap-mb 25 \
# --patience -1 \
# --required-batch-size-multiple 1 \
# --pooler-activation-fn tanh

