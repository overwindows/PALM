DATA_BIN=/datadrive/cnn_dm_bin

CUDA_VISIBLE_DEVICES=1 python -m train $DATA_BIN \
--user-dir src \
--min-loss-scale 0.0001 \
--model-parallel-size 1 \
--criterion label_smoothed_cross_entropy_with_masked_lm \
--optimizer adam \
--lr-scheduler polynomial_decay --total-num-update 100 \
--task auto_encoding_regressive \
--skip-invalid-size-inputs-valid-test \
--max-tokens 2048 \
--required-batch-size-multiple 1 \
--train-subset train \
--valid-subset valid \
--validate-interval 1 \
--max-tokens-valid 2048 \
--bucket-cap-mb 25 \
--arch palm_base \
--clip-norm 0.1 \
--update-freq [2] \
--lr [0.0006] \
--min-lr -1 \
--save-dir /datadrive/pretrain/fairseq/palm_en_small/logs \
--optimizer-overrides {} \
--keep-interval-updates -1 \
--keep-last-epochs -1 \
--keep-best-checkpoints -1 \
--no-epoch-checkpoints \
--best-checkpoint-metric loss \
--patience -1 \
--adam-betas "(0.9, 0.98)" \
--adam-eps 1e-06 \
--weight-decay 0.01 \
--warmup-updates 15 \
--power 1 \
--dropout 0.1 \
--attention-dropout 0.1 \
--num-segment 1 \
--pooler-activation-fn tanh --act-dropout 0.1

# --sample-break-mode complete_doc \
# --mask 0.3 \
# --mask-random 0.1 \
# --poisson-lambda 3.5 \
# --permute-sentences 1 \
# --mask-length span-poisson \
# --replace-length 1 \
# --cpu \
# --tensorboard-logdir /datadrive/pretrain/tests/fairseq/palm_en_small/logs \
# --max-update 100 \
# --restore-file checkpoint_last.pt \
# --save-interval 1 --save-interval-updates 50 \
# --share-all-embeddings \
# --layernorm-embedding \
# --share-decoder-input-output-embed \
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


