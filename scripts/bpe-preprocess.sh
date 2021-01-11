TASK=wikitext
# TASK=cnn_dm

for SPLIT in train val
do
  for LANG in source target
  do
    python -m utils.multiprocessing_bpe_encoder \
    --encoder-json gpt2_bpe/encoder.json \
    --vocab-bpe gpt2_bpe/vocab.bpe \
    --inputs "/datadrive/$TASK/$SPLIT.$LANG" \
    --outputs "/datadrive/$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
