TASK=summary
EXP=baseline
CORPUS=/apdcephfs/private_chewu/VLFM/video_summary/weibo_dataset

rm -rf ${CORPUS}_bin

fairseq-preprocess \
  --source-lang source \
  --target-lang target \
  --trainpref ${CORPUS}/train.bpe \
  --testpref ${CORPUS}/test.bpe \
  --destdir ${CORPUS}_bin/ \
  --workers 20 \
  --srcdict ../PALM/bpe/bert-base-multilingual-uncased-bpe-dict.txt \
  --joined-dictionary
