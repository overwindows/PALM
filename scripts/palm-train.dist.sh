total_updates=500000
warmup_updates=500      
# lr=3e-05
lr=0.001
max_tokens=2048
UPDATE_FREQ=4
pointer_layer=-2
# roberta_path=/bigdata/roberta.base/model.pt
datadrive=/apdcephfs/share_1351585/corpus
task=pretrain

root=/apdcephfs/private_chewu/PALM

rm -rf /apdcephfs/private_chewu/logdir/palm_"$task"

python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=2 --node_rank=0 --master_addr="9.134.252.217" \
    --master_port=12345 \
    --use_env /apdcephfs/private_chewu/PALM/utils/train.py "$datadrive"/"$task"_bin \
    --user-dir "$root"/src --truncate-source --source-lang source --target-lang target \
    --task auto_encoding_regressive --arch palm_base --criterion label_smoothed_cross_entropy_with_masked_lm --label-smoothing 0.1 \
    --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed \
    --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --lr-scheduler inverse_sqrt --lr "$lr" --max-update "$total_updates" --warmup-updates "$warmup_updates" \
    --act-dropout 0.1 --weight-decay 0.01 --dropout 0.1 --attention-dropout 0.1 --clip-norm 0.1 \
    --skip-invalid-size-inputs-valid-test --validate-interval 1 --max-tokens-valid 2048 --eval-bleu \
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
    --source-position-markers 1000 --update-freq "$UPDATE_FREQ" \
    --train-subset train \
    --valid-subset valid \
    --keep-interval-updates -1 \
    --keep-last-epochs -1 \
    --keep-best-checkpoints -1 \
    --no-epoch-checkpoints \
    --best-checkpoint-metric loss \
    --reset-optimizer --reset-dataloader --reset-meters \
    --wandb-project PALM \
    --ddp-backend=legacy_ddp \
    --save-dir /apdcephfs/private_chewu/palm_"${task}"_checkpoints \
    --tensorboard-logdir /apdcephfs/private_chewu/logdir/palm_"$task"
    # --restore-file $roberta_path \
    # --fp16 \
    # --find-unused-parameters \


# total_updates=200000
# warmup_updates=500      
# lr=3e-05
# #lr=0.001
# max_tokens=4096
# UPDATE_FREQ=4
# pointer_layer=-2
# roberta_path=/bigdata/roberta.base/model.pt
# EXP_NAME=palm_roberta

# CUDA_VISIBLE_DEVICES=1 python -m utils.train /datadrive/cnn_dm_bin \
#     --user-dir src \
#     --restore-file $roberta_path \
#     --max-tokens $max_tokens \
#     --task auto_encoding_regressive \
#     --source-lang source --target-lang target \
#     --truncate-source \
#     --layernorm-embedding \
#     --share-all-embeddings \
#     --share-decoder-input-output-embed \
#     --reset-optimizer --reset-dataloader --reset-meters \
#     --required-batch-size-multiple 1 \
#     --arch palm_base \
#     --criterion label_smoothed_cross_entropy_with_masked_lm --label-smoothing 0.1 \
#     --dropout 0.1 --attention-dropout 0.1 \
#     --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
#     --clip-norm 0.1 \
#     --eval-bleu \
#     --lr-scheduler inverse_sqrt --lr $lr --max-update $total_updates --warmup-updates $warmup_updates \
#     --update-freq $UPDATE_FREQ \
#     --skip-invalid-size-inputs-valid-test \
#     --alignment-layer "$pointer_layer" \
#     --alignment-heads 1 \
#     --encoder-normalize-before \
#     --decoder-normalize-before \
#     --find-unused-parameters \
#     --keep-interval-updates -1 \
#     --keep-last-epochs -1 \
#     --keep-best-checkpoints -1 \
#     --no-epoch-checkpoints \
#     --best-checkpoint-metric loss \
#     --act-dropout 0.1 --save-dir /bigdata/"$EXP_NAME"_checkpoints --tensorboard-logdir /bigdata/logdir/"$EXP_NAME"

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



