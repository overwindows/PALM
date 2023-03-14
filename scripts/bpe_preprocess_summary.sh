TASK=summary

# for SPLIT in train
for i in train test
do 
  echo $i
  for LANG in source target
  do
    python3 -m utils.multiprocessing_bpe_encoder \
    --inputs "/apdcephfs/private_chewu/VLFM/video_summary/weibo_dataset/$i.$LANG" \
    --outputs "/apdcephfs/private_chewu/VLFM/video_summary/weibo_dataset/$i.bpe.$LANG" \
    --workers 20 \
    --keep-empty;
    #mv /apdcephfs/private_kevinkyhong/corpus/train$i.$LANG /data/corpus/train$i.$LANG
  done
done
