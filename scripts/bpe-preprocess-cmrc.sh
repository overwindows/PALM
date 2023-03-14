TASK=pretrain_test

# for SPLIT in train
for i in train dev trial
do 
  echo $i
  for LANG in source target
  do
    python3 -m utils.multiprocessing_bpe_encoder \
    --inputs "/apdcephfs/private_kevinkyhong/data/cmrc2018_public/corpus/$i.$LANG" \
    --outputs "/apdcephfs/private_kevinkyhong/data/cmrc2018_public/corpus/$i.bpe.$LANG" \
    --workers 20 \
    --keep-empty;
    #mv /apdcephfs/private_kevinkyhong/corpus/train$i.$LANG /data/corpus/train$i.$LANG
  done
done
