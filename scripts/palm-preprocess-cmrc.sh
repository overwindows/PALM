TASK=pretrain
EXP=baseline
CORPUS=/apdcephfs/private_kevinkyhong/data/cmrc2018_public/corpus

rm -rf ${CORPUS}_bin

fairseq-preprocess \
  --source-lang source \
  --target-lang target \
  --trainpref ${CORPUS}/train.bpe \
  --validpref ${CORPUS}/dev.bpe \
  --testpref ${CORPUS}/trial.bpe \
  --destdir ${CORPUS}_bin/ \
  --workers 20 \
  --srcdict /apdcephfs/private_kevinkyhong/PALM/bpe/bert-base-multilingual-uncased-bpe-dict.txt \
  --joined-dictionary
