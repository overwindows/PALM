# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
PALM: Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation
"""
from typing import Optional
import math
import logging
from typing import Any, Dict, List, Optional, Tuple
from fairseq.models.fairseq_encoder import EncoderOut

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    register_model, register_model_architecture, FairseqModel, FairseqEncoder)
from fairseq.models.transformer import TransformerModel, TransformerDecoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules import (
    MultiheadAttention,
    PositionalEmbedding,
    FairseqDropout,
    LayerNorm,
    TransformerDecoderLayer,
    TransformerSentenceEncoder,
)
from fairseq.models.fairseq_incremental_decoder import FairseqIncrementalDecoder

from torch import Tensor
from .hub_interface import PALMHubInterface


logger = logging.getLogger(__name__)


@register_model("palm")
class PALMModel(TransformerModel):
    __jit_unused_properties__ = ["supported_targets"]

    @classmethod
    def hub_models(cls):
        return {
            "palm.base": None,
            "palm.large": None,
            "palm.large.mnli": None,
            "palm.large.cnn": None,
            "palm.large.xsum": None,
        }

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()
        if hasattr(self.encoder, "dictionary"):
            self.eos: int = self.encoder.dictionary.eos()

        self.copy_attention = args.copy_attention
        if self.copy_attention:
            self.copy_attn_layer = MultiheadAttention(
                args.decoder_embed_dim, args.copy_attention_heads, dropout=args.copy_attention_dropout, encoder_decoder_attention=True)
            self.copy_alpha_linear = nn.Linear(args.decoder_embed_dim, 1)

    def get_masked_targets(self, sample):
        """Get targets from either the sample or the net's output."""
        # print(sample.keys())
        # print(sample['masked_source'].size())
        return sample["masked_source"]

    @classmethod
    def build_model(cls, args, task):
        # set any default arguments
        palm_base_architecture(args)
        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample
        # print('Model max positions: {}'.format(args.max_positions))
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        if src_dict != tgt_dict:
            raise ValueError("PALM requires a joined dictionary")

        transformer_model = TransformerModel.build_model(args, task)

        encoder = PALMEncoder(args, src_dict)
        # encoder = transformer_model.encoder
        decoder = PALMDecoder(
            args, tgt_dict, transformer_model.decoder.embed_tokens)
        # decoder = transformer_model.decoder
        return PALMModel(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        super(PALMModel, PALMModel).add_args(parser)
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            help="Apply spectral normalization on the classification head",
        )

        parser.add_argument('--copy-attention', default=False, action='store_true',
                            help='train transformer decoder with copy attention')
        parser.add_argument('--copy-attention-heads', type=int, metavar='N', default=1,
                            help='num copy layer attention heads')
        parser.add_argument('--copy-attention-dropout', type=float, metavar='D', default=0.,
                            help='num copy layer attention dropout')
        parser.add_argument('--force-generation', type=float, metavar='P',
                            default=None,
                            help='set the vocabulary distribution weight to P, '
                                 'instead of predicting it from the input (1.0 '
                                 'corresponding to generation, 0.0 to pointing)')

    @property
    def supported_targets(self):
        return {"self"}

    def forward(
        self,
        src_tokens,
        src_lengths,
        masked_tokens=None,
        prev_output_tokens=None,
        features_only: bool = False,
        classification_head_name: Optional[str] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        segment_label=None,
    ):
        if classification_head_name is not None:
            features_only = True

        # PALM Encoder
        x, encoder_out = self.encoder(
            src_tokens, segment_label, masked_tokens
        )

        # Transformer Encoder
        # encoder_out = self.encoder(
        #     src_tokens,
        #     src_lengths=src_lengths,
        #     token_embeddings=token_embeddings,
        #     return_all_hiddens=return_all_hiddens
        # )

        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        eos: int = self.eos
        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(eos), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            for k, head in self.classification_heads.items():
                # for torch script only supports iteration
                if k == classification_head_name:
                    x = head(sentence_representation)
                    break
        return x, extra

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        sample_break_mode="eos",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            sample_break_mode=sample_break_mode,
            **kwargs,
        )
        return PALMHubInterface(x["args"], x["task"], x["models"][0])

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = PALMClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            do_spectral_norm=getattr(
                self.args, "spectral_norm_classification_head", False
            ),
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads."):].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(
                        head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
        if (
            loaded_dict_size == len(self.encoder.dictionary) + 1
            and "<mask>" not in self.encoder.dictionary
        ):
            truncate_emb("encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.output_projection.weight")
            truncate_emb("decoder.output_projection.weight")

        # When continued pretraining on new set of languages for mbart,
        # add extra lang embeddings at the end of embed_tokens.
        # Note: newly added languages are assumed to have been added at the end.
        if self.args.task == "multilingual_denoising" and loaded_dict_size < len(
            self.encoder.dictionary
        ):
            logger.info(
                "Adding extra language embeddings not found in pretrained model for "
                "continued pretraining of MBART on new set of languages."
            )
            loaded_mask_token_embedding = state_dict["encoder.embed_tokens.weight"][
                -1, :
            ]

            num_langids_to_add = len(
                self.encoder.dictionary) - loaded_dict_size
            embed_dim = state_dict["encoder.embed_tokens.weight"].size(1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            nn.init.normal_(new_lang_embed_to_add,
                            mean=0, std=embed_dim ** -0.5)
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict["encoder.embed_tokens.weight"].dtype,
            )

            state_dict["encoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["encoder.embed_tokens.weight"][
                        : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )
            state_dict["decoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["decoder.embed_tokens.weight"][
                        : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting", prefix +
                                "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


class PALMEncoder(FairseqEncoder):
    """
    Encoder for Masked Language Modelling.
    """

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        # print(args.act_dropout)
        self.padding_idx = dictionary.pad()
        self.vocab_size = dictionary.__len__()
        self.max_source_positions = args.max_positions
        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=self.padding_idx,
            vocab_size=self.vocab_size,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=self.max_source_positions,
            # num_segments=args.num_segment,
            use_position_embeddings=not args.no_token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_bert_init=args.apply_bert_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
        )

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        # self.sentence_projection_layer = None
        # self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(args, "remove_head", False)

        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.pooler_activation = utils.get_activation_fn(
            args.pooler_activation_fn)

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(
                torch.zeros(self.vocab_size))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim, self.vocab_size, bias=False
                )

            # if args.sent_loss:
            #     self.sentence_projection_layer = nn.Linear(
            #         args.encoder_embed_dim, self.sentence_out_dim, bias=False
            #     )

    def forward(self, src_tokens, segment_labels=None, masked_tokens=None, **unused):
        """
        Forward pass for Masked LM encoder. This first computes the token
        embedding using the token embedding matrix, position embeddings (if
        specified) and segment embeddings (if specified).

        Here we assume that the sentence representation corresponds to the
        output of the classification_token (see bert_task or cross_lingual_lm
        task for more details).
        Args:
            - src_tokens: B x T matrix representing sentences
            - segment_labels: B x T matrix representing segment label for tokens
        Returns:
            - a tuple of the following:
                - logits for predictions in format B x T x C to be used in
                  softmax afterwards
                - a dictionary of additional data, where 'pooled_output' contains
                  the representation for classification_token and 'inner_states'
                  is a list of internal model states used to compute the
                  predictions (similar in ELMO). 'sentence_logits'
                  is the prediction logit for NSP task and is only computed if
                  this is specified in the input arguments.
        """
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        inner_states, sentence_rep = self.sentence_encoder(
            src_tokens,
            segment_labels=segment_labels,
        )

        x = inner_states[-1].transpose(0, 1)

        # Keep original tokens
        encoder_out = self.layer_norm(self.activation_fn(
            self.lm_head_transform_weight(x)))

        # project masked tokens only
        if masked_tokens is not None:
            x = x[masked_tokens, :]

        x = self.layer_norm(self.activation_fn(
            self.lm_head_transform_weight(x)))

        pooled_output = self.pooler_activation(
            self.masked_lm_pooler(sentence_rep))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
            self.sentence_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.sentence_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias
        sentence_logits = None
        # if self.sentence_projection_layer:
        #     sentence_logits = self.sentence_projection_layer(pooled_output)

        return x, {
            "encoder_out": [encoder_out],
            "masked_out": x,
            "inner_states": inner_states,
            "pooled_output": pooled_output,
            # "sentence_logits": sentence_logits,
            "encoder_padding_mask": [encoder_padding_mask],
            "src_tokens": [src_tokens],  # B x T
        }

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_source_positions

    def upgrade_state_dict_named(self, state_dict, name):
        if isinstance(
            self.sentence_encoder.embed_positions, SinusoidalPositionalEmbedding
        ):
            state_dict[
                name + ".sentence_encoder.embed_positions._float_tensor"
            ] = torch.FloatTensor(1)
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if (
                    "embed_out.weight" in k
                    or "sentence_projection_layer.weight" in k
                    or "lm_output_learned_bias" in k
                ):
                    del state_dict[k]
        return state_dict


class PALMDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(
            embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(
            args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

        # Generation probabilities / interpolation coefficients are predicted
        # from the current decoder input embedding and the decoder output, which
        # is the size of output_embed_dim.
        p_gen_input_size = input_embed_dim + self.output_embed_dim
        self.project_p_gens = nn.Linear(p_gen_input_size, 1)
        nn.init.zeros_(self.project_p_gens.bias)

        # The dictionary may include a separate entry for an OOV token in each
        # input position, so that their identity can be restored from the
        # original source text.
        self.num_types = len(dictionary)
        # self.num_oov_types = args.source_position_markers
        self.num_oov_types = 0
        self.num_embeddings = self.num_types - self.num_oov_types
        self.force_p_gen = args.force_generation

        self.copy_attention = args.copy_attention
        self.attention_dropout = args.attention_dropout
        if self.copy_attention:
            self.copy_attn_layer = MultiheadAttention(
                embed_dim, args.copy_attention_heads, dropout=args.copy_attention_dropout)
            self.copy_alpha_linear = nn.Linear(embed_dim, 1)
            torch.nn.init.constant_(self.copy_alpha_linear.bias, 1.)
        self.classifier = nn.Linear(embed_dim, 2)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        assert not features_only
        if not features_only:
            # Embedding the tokens again for generation probability prediction,
            # so that we don't have to reimplement the whole extract_features()
            # method.
            if incremental_state is not None:
                prev_output_tokens = prev_output_tokens[:, -1:]
            prev_output_embed = self.embed_tokens(prev_output_tokens)
            prev_output_embed *= self.embed_scale
            predictors = torch.cat((prev_output_embed, x), 2)
            p_gens = self.project_p_gens(predictors)
            p_gens = torch.sigmoid(p_gens)
            x = self.output_layer(
                x, extra["attn"][0], encoder_out["src_tokens"][0], p_gens)

        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        copy_attn, copy_alpha = None, None
        if self.copy_attention:
            assert encoder_out is not None, \
                "--copy-attn can't be used with decoder only architecture"
            x_copy, copy_attn = self.copy_attn_layer(
                query=x,
                key=encoder_out['encoder_out'][0],
                value=encoder_out['encoder_out'][0],
                key_padding_mask=encoder_out['encoder_padding_mask'][0],
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=True,
            )
            x_copy = x_copy.transpose(0, 1)
            copy_alpha = torch.sigmoid(self.copy_alpha_linear(x_copy))
            attn = copy_attn  # use copy attn for alignment

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states, "encoder_out": encoder_out}

    # def output_layer(self, features):
    #     """Project features to the vocabulary size."""
    #     if self.adaptive_softmax is None:
    #         # project back to size of vocabulary
    #         return self.output_projection(features)
    #     else:
    #         return features

    def output_layer(self, features, attn, src_tokens, p_gens, **kwargs):
        """
        Project features to the vocabulary size and mix with the attention
        distributions.
        """
        if self.force_p_gen is not None:
            p_gens = self.force_p_gen

        # project back to size of vocabulary
        logits = super().output_layer(features, **kwargs)

        batch_size = logits.shape[0]
        output_length = logits.shape[1]
        assert logits.shape[2] == self.num_embeddings
        assert src_tokens.shape[0] == batch_size
        src_length = src_tokens.shape[1]

        # The final output distribution will be a mixture of the normal output
        # distribution (softmax of logits) and attention weights.
        gen_dists = super().get_normalized_probs(
            (logits, None), log_probs=False, sample=None
        )
        gen_dists = torch.mul(gen_dists, p_gens)
        padding_size = (batch_size, output_length, self.num_oov_types)
        padding = gen_dists.new_zeros(padding_size)
        gen_dists = torch.cat((gen_dists, padding), 2)
        assert gen_dists.shape[2] == self.num_types

        # Scatter attention distributions to distributions over the extended
        # vocabulary in a tensor of shape [batch_size, output_length,
        # vocab_size]. Each attention weight will be written into a location
        # that is for other dimensions the same as in the index tensor, but for
        # the third dimension it's the value of the index tensor (the token ID).
        attn = torch.mul(attn, 1 - p_gens)
        index = src_tokens[:, None, :]
        index = index.expand(batch_size, output_length, src_length)
        attn_dists_size = (batch_size, output_length, self.num_types)
        attn_dists = attn.new_zeros(attn_dists_size)
        attn_dists.scatter_add_(2, index, attn)

        # Final distributions, [batch_size, output_length, num_types].
        return gen_dists + attn_dists

    def get_normalized_probs(self, net_output, log_probs, sample):
        """
        Get normalized probabilities (or log probs) from a net's output.
        Pointer-generator network output is already normalized.
        """
        probs = net_output[0]
        # Make sure the probabilities are greater than zero when returning log
        # probabilities.
        return probs.clamp(1e-10, 1.0).log() if log_probs else probs

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(
                        name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


class PALMClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

        if do_spectral_norm:
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("palm", "palm_large")
def palm_large_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(
        args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(
        args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(
        args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(
        args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(
        args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

    args.copy_attention = getattr(args, 'copy_attention', False)
    args.copy_attention_heads = getattr(args, 'copy_attention_heads', 1)
    args.copy_attention_dropout = getattr(args, 'copy_attention_dropout', 0.)


@register_model_architecture("palm", "palm_base")
def palm_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(
        args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    palm_large_architecture(args)


@register_model_architecture("palm", "mpalm_large")
def mpalm_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    palm_large_architecture(args)


@register_model_architecture("palm", "mpalm_base")
def mpalm_base_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    palm_base_architecture(args)


@register_model_architecture("bart", "mpalm_base_wmt20")
def mpalm_base_wmt20_architecture(args):
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    mpalm_base_architecture(args)


# class TransformerPointerGeneratorDecoder(TransformerDecoder):
#     """
#     Transformer decoder consisting of *args.decoder_layers* layers. Each layer
#     is a :class:`TransformerDecoderLayer`. The pointer-generator variant mixes
#     the output probabilities with an attention distribution in the output layer.
#     Args:
#         args (argparse.Namespace): parsed command-line arguments
#         dictionary (~fairseq.data.Dictionary): decoding dictionary
#         embed_tokens (torch.nn.Embedding): output embedding
#     """

#     def __init__(self, args, dictionary, embed_tokens):
#         super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)

#         # In the pointer-generator model these arguments define the decoder
#         # layer and the number of attention heads that will be averaged to
#         # create the alignment for pointing.
#         self.alignment_heads = args.alignment_heads
#         self.alignment_layer = args.alignment_layer

#         input_embed_dim = embed_tokens.embedding_dim

#         # Generation probabilities / interpolation coefficients are predicted
#         # from the current decoder input embedding and the decoder output, which
#         # is the size of output_embed_dim.
#         p_gen_input_size = input_embed_dim + self.output_embed_dim
#         self.project_p_gens = nn.Linear(p_gen_input_size, 1)
#         nn.init.zeros_(self.project_p_gens.bias)

#         # The dictionary may include a separate entry for an OOV token in each
#         # input position, so that their identity can be restored from the
#         # original source text.
#         self.num_types = len(dictionary)
#         self.num_oov_types = args.source_position_markers
#         self.num_embeddings = self.num_types - self.num_oov_types
#         self.force_p_gen = args.force_generation

#     def forward(
#         self,
#         prev_output_tokens,
#         encoder_out: Optional[EncoderOut] = None,
#         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
#         features_only: bool = False,
#         alignment_layer: Optional[int] = 0,
#         alignment_heads: Optional[int] = 1,
#         src_lengths: Optional[Any] = None,
#         return_all_hiddens: bool = False,
#     ):
#         """
#         Args:
#             prev_output_tokens (LongTensor): previous decoder outputs of shape
#                 `(batch, tgt_len)`, for teacher forcing
#             encoder_out (EncoderOut, optional): output from the encoder, used
#                 for encoder-side attention
#             incremental_state (dict, optional): dictionary used for storing
#                 state during :ref:`Incremental decoding`
#             features_only (bool, optional): only return features without
#                 applying output layer (default: False)
#             alignment_layer (int, optional): 0-based index of the layer to be
#                 used for pointing (default: 0)
#             alignment_heads (int, optional): number of attention heads to be
#                 used for pointing (default: 1)
#         Returns:
#             tuple:
#                 - the decoder's output of shape `(batch, tgt_len, vocab)`
#                 - a dictionary with any model-specific outputs
#         """
#         # The normal Transformer model doesn't pass the alignment_layer and
#         # alignment_heads parameters correctly. We use our local variables.
#         x, extra = self.extract_features(
#             prev_output_tokens,
#             encoder_out=encoder_out,
#             incremental_state=incremental_state,
#             alignment_layer=self.alignment_layer,
#             alignment_heads=self.alignment_heads,
#         )
#         if not features_only:
#             # Embedding the tokens again for generation probability prediction,
#             # so that we don't have to reimplement the whole extract_features()
#             # method.
#             if incremental_state is not None:
#                 prev_output_tokens = prev_output_tokens[:, -1:]
#             prev_output_embed = self.embed_tokens(prev_output_tokens)
#             prev_output_embed *= self.embed_scale
#             predictors = torch.cat((prev_output_embed, x), 2)
#             p_gens = self.project_p_gens(predictors)
#             p_gens = torch.sigmoid(p_gens)
#             x = self.output_layer(x, extra["attn"][0], encoder_out["src_tokens"][0], p_gens)
#         return x, extra

#     def output_layer(self, features, attn, src_tokens, p_gens, **kwargs):
#         """
#         Project features to the vocabulary size and mix with the attention
#         distributions.
#         """
#         if self.force_p_gen is not None:
#             p_gens = self.force_p_gen

#         # project back to size of vocabulary
#         logits = super().output_layer(features, **kwargs)

#         batch_size = logits.shape[0]
#         output_length = logits.shape[1]
#         assert logits.shape[2] == self.num_embeddings
#         assert src_tokens.shape[0] == batch_size
#         src_length = src_tokens.shape[1]

#         # The final output distribution will be a mixture of the normal output
#         # distribution (softmax of logits) and attention weights.
#         gen_dists = super().get_normalized_probs(
#             (logits, None), log_probs=False, sample=None
#         )
#         gen_dists = torch.mul(gen_dists, p_gens)
#         padding_size = (batch_size, output_length, self.num_oov_types)
#         padding = gen_dists.new_zeros(padding_size)
#         gen_dists = torch.cat((gen_dists, padding), 2)
#         assert gen_dists.shape[2] == self.num_types

#         # Scatter attention distributions to distributions over the extended
#         # vocabulary in a tensor of shape [batch_size, output_length,
#         # vocab_size]. Each attention weight will be written into a location
#         # that is for other dimensions the same as in the index tensor, but for
#         # the third dimension it's the value of the index tensor (the token ID).
#         attn = torch.mul(attn, 1 - p_gens)
#         index = src_tokens[:, None, :]
#         index = index.expand(batch_size, output_length, src_length)
#         attn_dists_size = (batch_size, output_length, self.num_types)
#         attn_dists = attn.new_zeros(attn_dists_size)
#         attn_dists.scatter_add_(2, index, attn)

#         # Final distributions, [batch_size, output_length, num_types].
#         return gen_dists + attn_dists

#     def get_normalized_probs(self, net_output, log_probs, sample):
#         """
#         Get normalized probabilities (or log probs) from a net's output.
#         Pointer-generator network output is already normalized.
#         """
#         probs = net_output[0]
#         # Make sure the probabilities are greater than zero when returning log
#         # probabilities.
#         return probs.clamp(1e-10, 1.0).log() if log_probs else probs
