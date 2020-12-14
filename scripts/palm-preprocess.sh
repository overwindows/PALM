TASK=cnn_dm

fairseq-preprocess \
  --source-lang source \
  --target-lang target \
  --trainpref /datadrive/${TASK}/train.bpe \
  --validpref /datadrive/${TASK}/val.bpe \
  --destdir /datadrive/${TASK}_bin/ \
  --workers 60 \
  --srcdict dict.txt \
  --joined-dictionary
