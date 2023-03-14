export LANG=en_US.UTF-8 
fairseq-generate /apdcephfs/private_kevinkyhong/data/cmrc2018_public/corpus_bin/ \
--user-dir src \
--path /apdcephfs/private_kevinkyhong/fine_tuned_model/bart_finetuned_cmrc_checkpoints/checkpoint_best.pt \
--results-path /apdcephfs/private_kevinkyhong/fine_tuned_model/bart_finetuned_cmrc_checkpoints \
--skip-invalid-size-inputs-valid-test \
--max-tokens 2048 --required-batch-size-multiple 1 \
--valid-subset test \
--max-tokens-valid 2048 \
--batch-size 128 --beam 5 \
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
