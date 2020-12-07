# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import metrics, utils, modules
from fairseq.criterions import register_criterion

from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion("label_smoothed_cross_entropy_with_masked_lm")
class LabelSmoothedCrossEntropyCriterionWithMaskedLM(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, alignment_lambda):
        super().__init__(task, sentence_avg, label_smoothing)
        self.alignment_lambda = alignment_lambda

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument(
            "--alignment-lambda",
            default=0.05,
            type=float,
            metavar="D",
            help="weight for the alignment loss",
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(
            model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(
                0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        alignment_loss = None

        # Compute alignment loss only for training set and non dummy batches.
        if "alignments" in sample and sample["alignments"] is not None:
            alignment_loss = self.compute_alignment_loss(sample, net_output)

        if alignment_loss is not None:
            logging_output["alignment_loss"] = utils.item(alignment_loss.data)
            loss += self.alignment_lambda * alignment_loss

        masked_loss = self.compute_masked_loss(model, sample, net_output)
        loss += masked_loss

        return loss, sample_size, logging_output

    def compute_alignment_loss(self, sample, net_output):
        attn_prob = net_output[1]["attn"][0]
        bsz, tgt_sz, src_sz = attn_prob.shape
        attn = attn_prob.view(bsz * tgt_sz, src_sz)

        align = sample["alignments"]
        align_weights = sample["align_weights"].float()

        if len(align) > 0:
            # Alignment loss computation. align (shape [:, 2]) contains the src-tgt index pairs corresponding to
            # the alignments. align_weights (shape [:]) contains the 1 / frequency of a tgt index for normalizing.
            loss = -(
                (attn[align[:, 1][:, None], align[:, 0][:, None]]).log()
                * align_weights[:, None]
            ).sum()
        else:
            return None

        return loss

    def compute_masked_loss(self, model, sample, net_output):
        masked_tokens = sample["masked_source"].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).

        # if self.tpu:
        #    masked_tokens = None  # always project all tokens on TPU
        if masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        # logits = model(**sample["net_input"], masked_tokens=masked_tokens)[0]
        # print(net_output[1]["encoder_out"].keys())
        logits = net_output[1]["encoder_out"]['encoder_out'][0]
        targets = model.get_masked_targets(sample, [logits])
        
        if masked_tokens is not None:
            targets = targets[masked_tokens]
        if targets.size() != masked_tokens.size():
            print(targets.size(), logits.size(), masked_tokens.size())
            return .0
        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0)
                                  for log in logging_outputs))
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )
        alignment_loss_sum = utils.item(
            sum(log.get("alignment_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0)
                                 for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "alignment_loss",
            alignment_loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

# @register_criterion("masked_lm")
# class MaskedLmLoss(FairseqCriterion):
#     """
#     Implementation for the loss used in masked language model (MLM) training.
#     """

#     def __init__(self, task, tpu=False):
#         super().__init__(task)
#         self.tpu = tpu

#     def forward(self, model, sample, reduce=True):
#         """Compute the loss for the given sample.

#         Returns a tuple with three elements:
#         1) the loss
#         2) the sample size, which is used as the denominator for the gradient
#         3) logging outputs to display while training
#         """
#         masked_tokens = sample["target"].ne(self.padding_idx)
#         sample_size = masked_tokens.int().sum()

#         # Rare: when all tokens are masked, project all tokens.
#         # We use torch.where to avoid device-to-host transfers,
#         # except on CPU where torch.where is not well supported
#         # (see github.com/pytorch/pytorch/issues/26247).
#         if self.tpu:
#             masked_tokens = None  # always project all tokens on TPU
#         elif masked_tokens.device == torch.device("cpu"):
#             if not masked_tokens.any():
#                 masked_tokens = None
#         else:
#             masked_tokens = torch.where(
#                 masked_tokens.any(),
#                 masked_tokens,
#                 masked_tokens.new([True]),
#             )

#         logits = model(**sample["net_input"], masked_tokens=masked_tokens)[0]
#         targets = model.get_targets(sample, [logits])
#         if masked_tokens is not None:
#             targets = targets[masked_tokens]

#         loss = modules.cross_entropy(
#             logits.view(-1, logits.size(-1)),
#             targets.view(-1),
#             reduction="sum",
#             ignore_index=self.padding_idx,
#         )

#         logging_output = {
#             "loss": loss if self.tpu else loss.data,
#             "ntokens": sample["ntokens"],
#             "nsentences": sample["nsentences"],
#             "sample_size": sample_size,
#         }
#         return loss, sample_size, logging_output

#     @staticmethod
#     def reduce_metrics(logging_outputs) -> None:
#         """Aggregate logging outputs from data parallel training."""
#         loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
#         sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

#         metrics.log_scalar(
#             "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
#         )
#         metrics.log_derived(
#             "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
#         )

#     @staticmethod
#     def logging_outputs_can_be_summed() -> bool:
#         """
#         Whether the logging outputs returned by `forward` can be summed
#         across workers prior to calling `reduce_metrics`. Setting this
#         to True will improves distributed training speed.
#         """
#         return True
