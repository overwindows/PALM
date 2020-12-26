DATA_BIN=/datadrive/cnn_dm_bin
EXP_NAME=palm_trans
total_updates=200000
warmup_updates=500
lr=0.001
max_tokens=4096
update_freq=4
pointer_layer=-2

CUDA_VISIBLE_DEVICES=2 python -m utils.train $DATA_BIN \
--user-dir src --truncate-source --source-lang source --target-lang target \
--task auto_encoding_regressive --arch palm_base --criterion label_smoothed_cross_entropy_with_masked_lm --label-smoothing 0.1 \
--share-all-embeddings --share-decoder-input-output-embed --layernorm-embedding \
--optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
--lr-scheduler inverse_sqrt --lr "$lr" --max-update "$total_updates" --warmup-updates "$warmup_updates" \
--weight-decay 0.01 --dropout 0.1 --attention-dropout 0.1 --act-dropout 0.1 --clip-norm 0.1 \
--skip-invalid-size-inputs-valid-test \
--max-tokens "$max_tokens" --required-batch-size-multiple 1 \
--alignment-layer "$pointer_layer" \
--alignment-heads 1 \
--encoder-embed-dim 512 \
--encoder-ffn-embed-dim 2048 \
--encoder-attention-heads 8 \
--decoder-embed-dim 512 \
--decoder-ffn-embed-dim 2048 \
--decoder-attention-heads 8 \
--encoder-normalize-before \
--decoder-normalize-before \
--update-freq 4 --source-position-markers 1000 \
--train-subset train \
--valid-subset valid \
--validate-interval 1 \
--max-tokens-valid 2048 \
--keep-interval-updates -1 \
--keep-last-epochs -1 \
--keep-best-checkpoints -1 \
--no-epoch-checkpoints \
--best-checkpoint-metric loss \
--save-dir /bigdata/"$EXP_NAME"_checkpoints \
--tensorboard-logdir /bigdata/logdir/$EXP_NAME --reset-optimizer

# --sample-break-mode complete_doc \
# --mask 0.3 \
# --mask-random 0.1 \
# --poisson-lambda 3.5 \
# --permute-sentences 1 \
# --mask-length span-poisson \
# --replace-length 1 \
# --cpu \
# --max-update 100 \
# --restore-file checkpoint_last.pt \
# --save-interval 1 --save-interval-updates 50 \
# --encoder-layers 6 \
# --decoder-layers 6  \
# --encoder-learned-pos \
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
# --num-segment 1 \
# --min-lr -1 \
# --optimizer-overrides {} \
# --min-loss-scale 0.0001 \
# --power 1 \
# --bucket-cap-mb 25 \
# --patience -1 \
# --pooler-activation-fn tanh
# --tensorboard-logdir /bigdata/logdir/debug
# --eval-bleu \
# --reset-optimizer --reset-meters --reset-dataloader 