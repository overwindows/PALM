# TASK=cnn_dm
TASK=wikitext

rm -rf /datadrive/${TASK}_bin

fairseq-preprocess \
  --source-lang source \
  --target-lang target \
  --trainpref /datadrive/${TASK}/train.bpe \
  --validpref /datadrive/${TASK}/val.bpe \
  --destdir /datadrive/${TASK}_bin/ \
  --workers 60 \
  --srcdict gpt2_bpe/dict.txt \
  --joined-dictionary
