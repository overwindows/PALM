TASK=wikitext
# TASK=cnn_dm

for SPLIT in train val
do
  for LANG in source target
  do
    python3 -m utils.multiprocessing_bpe_encoder \
    --inputs "/corpus/$TASK/$SPLIT.$LANG" \
    --outputs "/corpus/$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done