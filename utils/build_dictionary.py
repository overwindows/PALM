import os
import sys
import logging
import tqdm
import glob
import argparse


logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=os.environ.get("LOGLEVEL", "INFO").upper(),
                    stream=sys.stdout,
                    )
logger = logging.getLogger(__name__)


def read_vocab(vocab_file: str):
    token_cnt = {}
    with open(vocab_file, 'r', encoding="utf-8") as fin:
        for token in tqdm.tqdm(fin):
            token = token.strip()
            token_cnt[token] = 0
    return token_cnt


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # # parser.add_argument(
    # #     "--encoder-json",
    # #     help="path to encoder.json",
    # # )
    # parser.add_argument(
    #     "--vocab",
    #     type=str,
    #     help="path to vocab",
    #     default='/apdcephfs/private_chewu/PALM/bert-base-multilingual-uncased-bpe/vocab.txt'
    # )
    # parser.add_argument(
    #     "--inputs",
    #     nargs="+",
    #     default=["-"],
    #     help="input files to filter/encode",
    # )
    # parser.add_argument(
    #     "--outputs",
    #     nargs="+",
    #     default=["-"],
    #     help="path to save encoded outputs",
    # )
    # parser.add_argument(
    #     "--keep-empty",
    #     action="store_true",
    #     help="keep empty lines",
    # )
    # parser.add_argument("--workers", type=int, default=20)
    # args = parser.parse_args()

    vocab_file = '/apdcephfs/private_chewu/PALM/bert-base-multilingual-uncased-bpe/vocab.txt'
    bpe_files_dir = "/apdcephfs/private_chewu/corpus/pretrain"
    dict_file = '/apdcephfs/private_chewu/PALM/bert-base-multilingual-uncased-bpe/dict.txt'

    vocab = read_vocab(vocab_file)

    bpe_files = glob.glob(os.path.join(bpe_files_dir, '*.bpe.*'))

    for file in bpe_files:
        file_path = os.path.join(bpe_files_dir, file)
        logger.info('loading {}'.format(file_path))

        with open(file_path, 'r') as fin:
            for line in tqdm.tqdm(fin.readlines()):
                tokens = line.split()
                for t in tokens:
                    assert t in vocab
                    vocab[t] += 1

    with open('/apdcephfs/private_chewu/PALM/bert-base-multilingual-uncased-bpe/dict.txt', 'w') as fout:
        for k, v in vocab.items():
            fout.write('{} {}\n'.format(k, v))

    # d = {}
    # with open('/apdcephfs/private_chewu/PALM/bert-base-multilingual-uncased-bpe/dict_raw.txt', 'r', encoding="utf-8") as fin:
    #     for line in tqdm.tqdm(fin.readlines()):
    #         k, v = line.strip('\t')
    #         d[k] = v

    # with open(dict_file, 'w') as fout:
    #     with open(vocab_file, 'r', encoding="utf-8") as fin:
    #         for token in tqdm.tqdm(fin):
    #             token = token.strip()
    #             if token in d:
    #                 fout.write('{}\t{}\n'.format(token, d[token]))
    #             else:
    #                 fout.write('{}\t{}\n'.format(token, 0))
    # # p = Process(target=build_data, args=(
    #     out_prefix, raw_text_lines, seg_ix))

    # while len(procs) > 1:
    #     time.sleep(0.5)

    # p.start()
    # procs.append(p)
    # build_data(out_prefix, raw_text_lines, seg_ix)
    # seg_ix += 1
    # raw_text_lines = None