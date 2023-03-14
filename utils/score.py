#!/usr/bin/env python3
"""
BLEU / ROUGE scoring of generated translations against reference translations.
"""

import argparse
import os
import sys

from fairseq.data import dictionary
from fairseq.scoring import bleu
import rouge


def get_parser():
    parser = argparse.ArgumentParser(
        description="Command-line script for BLEU scoring."
    )
    # fmt: off
    parser.add_argument('-s', '--sys', default='-', help='system output')
    # parser.add_argument('-r', '--ref', required=True, help='references')
    parser.add_argument('-o', '--order', default=4, metavar='N',
                        type=int, help='consider ngrams up to this order')
    parser.add_argument('--ignore-case', action='store_true',
                        help='case-insensitive scoring')
    parser.add_argument('--rouge', action='store_true', help='score with rouge')
    parser.add_argument("--gen_file", default=None, type=str,
                        help="Path of the generated file.")
    parser.add_argument('--sacrebleu', action='store_true',
                        help='score with sacrebleu')
    parser.add_argument('--sentence-bleu', action='store_true',
                        help='report sentence-level BLEUs (i.e., with +1 smoothing)')
    # fmt: on
    return parser


def cli_main(args):

    print(args)

    assert args.sys == "-" or os.path.exists(
        args.sys
    ), "System output file {} does not exist".format(args.sys)
    assert os.path.exists(
        args.ref), "Reference file {} does not exist".format(args.ref)

    dict = dictionary.Dictionary()

    def readlines(fd):
        for line in fd.readlines():
            if args.ignore_case:
                yield line.lower()
            else:
                yield line

    if args.sacrebleu:
        import sacrebleu

        def score(fdsys):
            with open(args.ref) as fdref:
                print(sacrebleu.corpus_bleu(fdsys, [fdref]).format())

    elif args.sentence_bleu:

        def score(fdsys):
            with open(args.ref) as fdref:
                scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())
                for i, (sys_tok, ref_tok) in enumerate(
                    zip(readlines(fdsys), readlines(fdref))
                ):
                    scorer.reset(one_init=True)
                    sys_tok = dict.encode_line(sys_tok)
                    ref_tok = dict.encode_line(ref_tok)
                    scorer.add(ref_tok, sys_tok)
                    print(i, scorer.result_string(args.order))

    else:

        def score(fdsys):
            with open(args.ref) as fdref:
                scorer = bleu.Scorer(
                    bleu.BleuConfig(
                        pad=dict.pad(),
                        eos=dict.eos(),
                        unk=dict.unk(),
                    )
                )
                for sys_tok, ref_tok in zip(readlines(fdsys), readlines(fdref)):
                    sys_tok = dict.encode_line(sys_tok)
                    ref_tok = dict.encode_line(ref_tok)
                    scorer.add(ref_tok, sys_tok)
                print(scorer.result_string(args.order))

    if args.sys == "-":
        score(sys.stdin)
    else:
        with open(args.sys, "r") as f:
            score(f)


def rouge_main(args):
    rouge_evaluator = rouge.Rouge()
    refs = []
    hyps = []

    with open(args.gen_file, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            if line.startswith('T-'):
                head, text = line.strip().split('\t')
                refs.append(text)
            if line.startswith('H-'):
                head, _, text = line.strip().split('\t')
                hyps.append(text)

    assert len(refs) == len(hyps)

    avg_score = rouge_evaluator.get_scores(
        hyps, refs, avg=True)
    rouge_l_fscore = avg_score['rouge-l']['f']
    print('Samples: {}'.format(len(refs)))
    print('Rouge-L: {}'.format(rouge_l_fscore))
    print('Rouge-1: {}'.format(avg_score['rouge-1']['f']))
    print('Rouge-2: {}'.format(avg_score['rouge-2']['f']))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.rouge:
        rouge_main(args)
    else:
        cli_main(args)
