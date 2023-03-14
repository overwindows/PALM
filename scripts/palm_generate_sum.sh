export LANG=en_US.UTF-8 

# Note! It's a Tanslation Task.
# fairseq-generate /apdcephfs/private_chewu/VLFM/video_summary/weibo_dataset_bin/ \
python3 -m utils.generate /apdcephfs/private_chewu/VLFM/video_summary/weibo_dataset_bin/ \
--user-dir src \
--path /apdcephfs/private_chewu/models/palm_finetune_weibo_checkpoints/checkpoint_best.pt \
--results-path /apdcephfs/private_chewu/VLFM/video_summary/weibo_dataset \
--skip-invalid-size-inputs-valid-test \
--max-tokens 2048 --required-batch-size-multiple 1 \
--valid-subset test \
--max-tokens-valid 2048 \
--batch-size 128 --beam 5 \
# --task auto_encoding_regressive
#datadrive=/apdcephfs/private_kevinkyhong/data/cmrc2018_public/corpus_bin
#model_path=/apdcephfs/private_kevinkyhong/fine_tuned_model/palm_finetune_cmrc_checkpoints/checkpoint_best.pt

#CUDA_VISIBLE_DEVICES=0 python3 -m utils.generate "$datadrive" \
#--user-dir src --truncate-source --source-lang source --target-lang target \
#--task auto_encoding_regressive --criterion label_smoothed_cross_entropy_with_masked_lm \
#--skip-invalid-size-inputs-valid-test \
#--max-tokens 2048 --required-batch-size-multiple 1 \
#--valid-subset valid \
#--max-tokens-valid 2048 \
#--path $model_path \
#--batch-size 128 --beam 5

# CUDA_VISIBLE_DEVICES=0 python3 -m utils.generate "$datadrive" \
#--user-dir src \
#--task auto_encoding_regressive --criterion label_smoothed_cross_entropy_with_masked_lm \
#--path $model_path \
#--batch-size 128 --beam 5
