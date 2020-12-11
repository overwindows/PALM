# PALM: Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation (EMNLP 2020)
## CNN-DM Data
- [CNN Stories](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ)
- [DailyMail Stories](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs)
<!--## Url lists
```
wget https://raw.githubusercontent.com/artmatsak/cnn-dailymail/master/url_lists/all_train.txt
wget https://raw.githubusercontent.com/artmatsak/cnn-dailymail/master/url_lists/all_test.txt
wget https://raw.githubusercontent.com/artmatsak/cnn-dailymail/master/url_lists/all_val.txt
```
-->
## Make datafiles
```
python -m make_datafiles /datadrive/cnn/stories /datadrive/dailymail/stories
```
## BPE preprocess
```
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

sh bpe_preprocess.sh
```

## Binarize dataset
```
sh preprocess.sh
```

## Model Training
```
sh palm-train.sh
```

## Reference
- [CNN-DM Data](https://github.com/abisee/cnn-dailymail)