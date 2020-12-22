# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
PALM: Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation
"""

from typing import Optional
import math
import logging
from typing import Any, Dict, List, Optional
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    gen_parser_from_dataclass,
)
from fairseq.models import (
    register_model, register_model_architecture)
from fairseq.models.transformer import TransformerModel, TransformerDecoder, TransformerEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules import (
    PositionalEmbedding,
    FairseqDropout,
    LayerNorm,
    TransformerDecoderLayer,
    SinusoidalPositionalEmbedding,
)
from fairseq.checkpoint_utils import prune_state_dict
from omegaconf import DictConfig

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

        if hasattr(self.encoder, "dictionary"):
            self.eos: int = self.encoder.dictionary.eos()

    @classmethod
    def build_model(cls, args, task):
        # set any default arguments
        palm_base_architecture(args)
        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        if src_dict != tgt_dict:
            raise ValueError("PALM requires a joined dictionary")

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError(
                    "--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        assert encoder_embed_tokens == decoder_embed_tokens

        encoder = PALMEncoder(args, src_dict, encoder_embed_tokens)
        decoder = PALMDecoder(args, tgt_dict, decoder_embed_tokens)

        return PALMModel(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)

        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )

        parser.add_argument(
            "--tokens-per-sample",
            default=1024,
            type=int,
            help="max number of total tokens over all segments "
            "per sample for BERT dataset",
        )

        parser.add_argument('--alignment-heads', type=int, metavar='N',
                            help='number of attention heads to be used for '
                                 'pointing')
        parser.add_argument('--alignment-layer', type=int, metavar='I',
                            help='layer number to be used for pointing (0 '
                                 'corresponding to the bottommost layer)')
        parser.add_argument('--source-position-markers', type=int, metavar='N',
                            help='dictionary includes N additional items that '
                                 'represent an OOV token at a particular input '
                                 'position')
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
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):

        # PALM Encoder
        encoder_out = self.encoder(
            src_tokens, src_lengths, masked_tokens
        )

        # PALM Decoder
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

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

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """

        if model_cfg is None and args is not None:
            logger.warn("using 'args' is deprecated, please update your code to use dataclass config")
            model_cfg = convert_namespace_to_omegaconf(args).model

        self.upgrade_state_dict(state_dict)
        new_state_dict = prune_state_dict(state_dict, model_cfg)
        return super().load_state_dict(new_state_dict, False)

    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        self.upgrade_state_dict_named(state_dict, "")
    
    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # For fairseq roberta model
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder.sentence_encoder") :]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        super().upgrade_state_dict_named(state_dict, name)

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
        print(state_dict.keys())
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(
            0)
        if (
            loaded_dict_size == len(self.encoder.dictionary) + 1
            and "<mask>" not in self.encoder.dictionary
        ):
            truncate_emb("encoder.sentence_encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.sentence_encoder.output_projection.weight")
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
            loaded_mask_token_embedding = state_dict["encoder.sentence_encoder.embed_tokens.weight"][
                -1, :
            ]

            num_langids_to_add = len(
                self.encoder.dictionary) - loaded_dict_size
            embed_dim = state_dict["encoder.sentence_encoder.embed_tokens.weight"].size(
                1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            nn.init.normal_(new_lang_embed_to_add,
                            mean=0, std=embed_dim ** -0.5)
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict["encoder.sentence_encoder.embed_tokens.weight"].dtype,
            )

            state_dict["encoder.sentence_encoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["encoder.sentence_encoder.embed_tokens.weight"][
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

        # In the pointer-generator model these arguments define the decoder
        # layer and the number of attention heads that will be averaged to
        # create the alignment for pointing.
        self.alignment_heads = args.alignment_heads
        self.alignment_layer = args.alignment_layer

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

        # self.classifier = nn.Linear(embed_dim, 2)

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
        x, extra = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

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
            assert encoder_out is not None and len(
                encoder_out["encoder_out"]) > 0
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

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states, "masked_encoder_out": encoder_out['masked_encoder_out']}

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
        assert src_tokens.shape[0] == batch_size, (
            batch_size, src_tokens.size())
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


class PALMEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`. The pointer-generator variant adds
    the source tokens to the encoder output as these are otherwise not passed
    to the decoder.
    """

    def forward(self, src_tokens, src_lengths, masked_tokens = None, **kwargs):
        """
        Runs the `forward()` method of the parent Transformer class. Then adds
        the source tokens into the encoder output tuple.
        While it might be more elegant that the model would pass the source
        tokens to the `forward()` method of the decoder too, this would require
        changes to `SequenceGenerator`.
        Args:
            src_tokens (torch.LongTensor): tokens in the source language of
                shape `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
                - **src_tokens** (Tensor): input token ids of shape
                  `(batch, src_len)`
        """
        encoder_out = super().forward(src_tokens, src_lengths, **kwargs)

        masked_encoder_out = None
        # project masked tokens
        if masked_tokens is not None:
            x = encoder_out["encoder_out"][0].transpose(0, 1)
            x = x[masked_tokens, :]
            # project back to size of vocabulary, self.share_input_output_embed and hasattr(self.embed_tokens, "weight")
            masked_encoder_out = F.linear(x, self.embed_tokens.weight)

        return {
            "encoder_out": encoder_out["encoder_out"],  # T x B x C
            "encoder_padding_mask": encoder_out["encoder_padding_mask"],
            "encoder_embedding": encoder_out["encoder_embedding"],  # B x T x C
            "encoder_states": encoder_out["encoder_states"],  # List[T x B x C]
            "src_tokens": [src_tokens],  # B x T
            "src_lengths": [],
            'masked_encoder_out': [masked_encoder_out] if masked_tokens is not None else [],  # B x T
        }
    
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"]
                            [0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(
                    0, new_order)
            ]
        if len(encoder_out["masked_encoder_out"]) == 0:
            new_masked_out = []
        else:
            new_masked_out = [encoder_out["masked_encoder_out"][0].index_select(0, new_order)]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]
                        ).index_select(0, new_order)]
        # if len(encoder_out["src_lengths"]) == 0:
        #     src_lengths = []
        # else:
        #     src_lengths = [(encoder_out["src_lengths"][0]
        #                     ).index_select(0, new_order)]

        # encoder_states = encoder_out["encoder_states"]
        # if len(encoder_states) > 0:
        #     for idx, state in enumerate(encoder_states):
        #         encoder_states[idx] = state.index_select(1, new_order)
        # print(new_encoder_out[0].size(), new_encoder_padding_mask[0].size())
        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "masked_encoder_out": new_masked_out,
            # "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            # "src_lengths": src_lengths,  # B x 1
        }


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
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
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
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
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

    args.adaptive_input = getattr(args, "adaptive_input", False)

    args.alignment_heads = getattr(args, "alignment_heads", 1)
    args.alignment_layer = getattr(args, "alignment_layer", -1)
    if args.alignment_layer < 0:
        args.alignment_layer = args.decoder_layers + args.alignment_layer


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


# class PALMClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""

#     def __init__(
#         self,
#         input_dim,
#         inner_dim,
#         num_classes,
#         activation_fn,
#         pooler_dropout,
#         do_spectral_norm=False,
#     ):
#         super().__init__()
#         self.dense = nn.Linear(input_dim, inner_dim)
#         self.activation_fn = utils.get_activation_fn(activation_fn)
#         self.dropout = nn.Dropout(p=pooler_dropout)
#         self.out_proj = nn.Linear(inner_dim, num_classes)

#         if do_spectral_norm:
#             self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

#     def forward(self, features, **kwargs):
#         x = features
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = self.activation_fn(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x


# class PALMEncoder(FairseqEncoder):
#     """
#     Encoder for Masked Language Modelling.
#     """

#     def __init__(self, args, dictionary, embed_tokens):
#         super().__init__(dictionary)
#         self.padding_idx = dictionary.pad()
#         self.vocab_size = dictionary.__len__()
#         self.max_source_positions = args.max_positions
#         self.sentence_encoder = TransformerSentenceEncoder(
#             padding_idx=self.padding_idx,
#             vocab_size=self.vocab_size,
#             num_encoder_layers=args.encoder_layers,
#             embedding_dim=args.encoder_embed_dim,
#             ffn_embedding_dim=args.encoder_ffn_embed_dim,
#             num_attention_heads=args.encoder_attention_heads,
#             dropout=args.dropout,
#             attention_dropout=args.attention_dropout,
#             activation_dropout=args.act_dropout,
#             max_seq_len=self.max_source_positions,
#             # num_segments=args.num_segment,
#             use_position_embeddings=not args.no_token_positional_embeddings,
#             encoder_normalize_before=args.encoder_normalize_before,
#             apply_bert_init=args.apply_bert_init,
#             activation_fn=args.activation_fn,
#             learned_pos_embedding=args.encoder_learned_pos,
#         )

#         self.share_input_output_embed = args.share_encoder_input_output_embed
#         self.embed_out = None
#         # self.sentence_projection_layer = None
#         # self.sentence_out_dim = args.sentence_class_num
#         self.lm_output_learned_bias = None

#         # Remove head is set to true during fine-tuning
#         self.load_softmax = not getattr(args, "remove_head", False)

#         self.masked_lm_pooler = nn.Linear(
#             args.encoder_embed_dim, args.encoder_embed_dim
#         )
#         self.pooler_activation = utils.get_activation_fn(
#             args.pooler_activation_fn)

#         self.lm_head_transform_weight = nn.Linear(
#             args.encoder_embed_dim, args.encoder_embed_dim
#         )
#         self.activation_fn = utils.get_activation_fn(args.activation_fn)
#         self.layer_norm = LayerNorm(args.encoder_embed_dim)

#         self.lm_output_learned_bias = None
#         if self.load_softmax:
#             self.lm_output_learned_bias = nn.Parameter(
#                 torch.zeros(self.vocab_size))

#             if not self.share_input_output_embed:
#                 self.embed_out = nn.Linear(
#                     args.encoder_embed_dim, self.vocab_size, bias=False
#                 )

#             # if args.sent_loss:
#             #     self.sentence_projection_layer = nn.Linear(
#             #         args.encoder_embed_dim, self.sentence_out_dim, bias=False
#             #     )

#     def forward(self, src_tokens, segment_labels=None, masked_tokens=None, **unused):
#         """
#         Forward pass for Masked LM encoder. This first computes the token
#         embedding using the token embedding matrix, position embeddings (if
#         specified) and segment embeddings (if specified).

#         Here we assume that the sentence representation corresponds to the
#         output of the classification_token (see bert_task or cross_lingual_lm
#         task for more details).
#         Args:
#             - src_tokens: B x T matrix representing sentences
#             - segment_labels: B x T matrix representing segment label for tokens
#         Returns:
#             - a tuple of the following:
#                 - logits for predictions in format B x T x C to be used in
#                   softmax afterwards
#                 - a dictionary of additional data, where 'pooled_output' contains
#                   the representation for classification_token and 'inner_states'
#                   is a list of internal model states used to compute the
#                   predictions (similar in ELMO). 'sentence_logits'
#                   is the prediction logit for NSP task and is only computed if
#                   this is specified in the input arguments.
#         """
#         encoder_padding_mask = src_tokens.eq(self.padding_idx)
#         inner_states, sentence_rep = self.sentence_encoder(
#             src_tokens,
#             segment_labels=segment_labels,
#         )
#         x = inner_states[-1].transpose(0, 1)

#         # Keep original tokens
#         encoder_out = self.layer_norm(self.activation_fn(
#             self.lm_head_transform_weight(x)))

#         # project masked tokens only
#         if masked_tokens is not None:
#             x = x[masked_tokens, :]

#         x = self.layer_norm(self.activation_fn(
#             self.lm_head_transform_weight(x)))

#         # pooled_output = self.pooler_activation(
#         #     self.masked_lm_pooler(sentence_rep))

#         # project back to size of vocabulary
#         if self.share_input_output_embed and hasattr(
#             self.sentence_encoder.embed_tokens, "weight"
#         ):
#             x = F.linear(x, self.sentence_encoder.embed_tokens.weight)
#         elif self.embed_out is not None:
#             x = self.embed_out(x)
#         if self.lm_output_learned_bias is not None:
#             x = x + self.lm_output_learned_bias
#         # sentence_logits = None
#         # if self.sentence_projection_layer:
#         #     sentence_logits = self.sentence_projection_layer(pooled_output)

#         return {
#             "encoder_out": [encoder_out],
#             "masked_out": x,
#             "inner_states": inner_states,
#             # "pooled_output": pooled_output,
#             # "sentence_logits": sentence_logits,
#             "encoder_padding_mask": [encoder_padding_mask],
#             "src_tokens": [src_tokens],  # B x T
#         }



#     def max_positions(self):
#         """Maximum output length supported by the encoder."""
#         return self.max_source_positions

#     def upgrade_state_dict_named(self, state_dict, name):
#         if isinstance(
#             self.sentence_encoder.embed_positions, SinusoidalPositionalEmbedding
#         ):
#             state_dict[
#                 name + ".sentence_encoder.embed_positions._float_tensor"
#             ] = torch.FloatTensor(1)
#         if not self.load_softmax:
#             for k in list(state_dict.keys()):
#                 if (
#                     "embed_out.weight" in k
#                     or "sentence_projection_layer.weight" in k
#                     or "lm_output_learned_bias" in k
#                 ):
#                     del state_dict[k]
#         return state_dict
