# TASK=cnn_dm
TASK=wikitext

rm -rf /corpus/${TASK}_bin

fairseq-preprocess \
  --source-lang source \
  --target-lang target \
  --trainpref /corpus/${TASK}/train.bpe \
  --validpref /corpus/${TASK}/val.bpe \
  --destdir /corpus/${TASK}_bin/ \
  --workers 60 \
  --srcdict gpt2_bpe/dict.txt \
  --joined-dictionary
