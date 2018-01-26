# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
# This file contains a custom ByteNet implementation

import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from fairseq.data import LanguagePairDataset
from fairseq.modules import BeamableMM
from fairseq.modules import LearnedPositionalEmbedding
from fairseq.modules import ResBlock
from fairseq.modules import GradMultiply, LinearizedConvolution

from . import FairseqEncoder, FairseqIncrementalDecoder, FairseqModel


class BNModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        #  self.encoder.num_attention_layers = sum(
        #      layer is not None for layer in decoder.attention)


class BNEncoder(FairseqEncoder):
    """Convolutional encoder"""
    def __init__(self, dictionary, embed_dim=512, max_positions=1024,
                 convolutions=((512, 3),) * 20, dropout=0.1):
        super().__init__(dictionary)
        self.dropout = dropout
        self.num_attention_layers = None

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_positions = PositionalEmbedding(
            max_positions, embed_dim, padding_idx,
            left_pad=LanguagePairDataset.LEFT_PAD_SOURCE)

        in_channels = convolutions[0][1]
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)

        self.resblocks = nn.ModuleList()
        for en_set in convolutions:
            # for each set
            (dil_rates, num_channels, kernel_size, causal) = en_set
            for dil_rate in dil_rates:
                self.resblocks.append(
                    ResBlock(in_channels=num_channels,
                             kernel_size=kernel_size,
                             dilation=dil_rate,
                             causal=causal))

        self.fc2 = Linear(in_channels, embed_dim)

    def forward(self, src_tokens):
        # embed tokens and positions
        x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # temporal convolutions
        for resblock in self.resblocks:
            x = resblock(x)

        # B x C x T -> B x T x C
        x = x.transpose(1, 2)

        # project back to size of embedding
        x = self.fc2(x)

        # scale gradients (this only affects backward, not forward)
        #  x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        return x, y

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim, conv_channels)

        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = self.bmm(x, encoder_out[0])

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = self.bmm(x, encoder_out[1])

        # scale attention output
        s = encoder_out[1].size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        """Replace torch.bmm with BeamableMM."""
        if beamable_mm_beam_size is not None:
            del self.bmm
            self.add_module('bmm', BeamableMM(beamable_mm_beam_size))


class BNDecoder(FairseqIncrementalDecoder):
    """Convolutional decoder"""
    def __init__(self, dictionary, embed_dim=512, out_embed_dim=256,
                 max_positions=1024, convolutions=((512, 3),) * 20,
                 attention=True, dropout=0.1, share_embed=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([2]))
        self.dropout = dropout

        in_channels = convolutions[0][1]
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [False] * len(convolutions[0][0])
        if not isinstance(attention, list) or \
                len(attention) != len(convolutions[0][0]):
            raise ValueError(
                'Attention is expected to be a list of booleans of '
                'length equal to the number of layers.')

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_positions = PositionalEmbedding(
            max_positions, embed_dim, padding_idx,
            left_pad=LanguagePairDataset.LEFT_PAD_TARGET)

        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)

        self.resblocks = nn.ModuleList()
        self.attention = nn.ModuleList()
        for en_set in convolutions:
            # for each set
            (dil_rates, num_channels, kernel_size, causal) = en_set
            # define residual layer
            for i, dil_rate in enumerate(dil_rates):
                self.resblocks.append(
                    ResBlock(in_channels=num_channels,
                             kernel_size=kernel_size,
                             dilation=dil_rate,
                             causal=causal,
                             mode="decoder"))
                self.attention.append(
                    AttentionLayer(num_channels, embed_dim)
                    if attention[i] else None)

        self.fc2 = Linear(in_channels, out_embed_dim)
        if share_embed:
            assert out_embed_dim == embed_dim, \
                "Shared embed weights implies same dimensions " \
                " out_embed_dim={} vs embed_dim={}".format(
                    out_embed_dim, embed_dim)
            self.fc3 = nn.Linear(out_embed_dim, num_embeddings)
            self.fc3.weight = self.embed_tokens.weight
        else:
            self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)

    def forward(self, input_tokens, encoder_out):
        # split and transpose encoder outputs
        encoder_a, encoder_b = self._split_encoder_out(encoder_out)

        # embed positions
        positions = self.embed_positions(input_tokens)

        if self._is_incremental_eval:
            # keep only the last token for incremental forward pass
            input_tokens = input_tokens[:, -1:]

        # embed tokens and positions
        x = self.embed_tokens(input_tokens) + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> B x C x T
        x = self._transpose_unless_incremental_eval(x)
        print (x.size())

        # temporal convolutions
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        for resblock, attention in zip(self.resblocks, self.attention):
            residual = x
            x = resblock(x)

            if attention is not None:
                x = self._transpose_unless_incremental_eval(x)

                x, attn_scores = attention(
                    x, target_embedding, (encoder_a, encoder_b))
                attn_scores = attn_scores / num_attn_layers
                if avg_attn_scores is None:
                    avg_attn_scores = attn_scores
                else:
                    avg_attn_scores.add_(attn_scores)

                x = self._transpose_unless_incremental_eval(x)

            x = x+residual

        # B x C x T -> B x T x C
        x = self._transpose_unless_incremental_eval(x)

        # project back to size of vocabulary
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x, avg_attn_scores

    def reorder_incremental_state(self, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        super().reorder_incremental_state(new_order)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if state_dict.get('decoder.version', torch.Tensor([1]))[0] < 2:
            # old models use incorrect weight norm dimension
            for i, conv in enumerate(self.resblocks):
                # reconfigure weight norm
                nn.utils.remove_weight_norm(conv)
                self.resblocks[i] = nn.utils.weight_norm(conv, dim=0)
            state_dict['decoder.version'] = torch.Tensor([1])
        return state_dict

    def _split_encoder_out(self, encoder_out):
        """Split and transpose encoder outputs.

        This is cached when doing incremental inference.
        """
        cached_result = self.get_incremental_state('encoder_out')
        if cached_result:
            return cached_result

        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        encoder_a = encoder_a.transpose(1, 2).contiguous()
        result = (encoder_a, encoder_b)

        return self.set_incremental_state('encoder_out', result)

    def _transpose_unless_incremental_eval(self, x):
        #  if self._is_incremental_eval:
        #      return x
        return x.transpose(1, 2)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad):
    m = LearnedPositionalEmbedding(
        num_embeddings, embedding_dim, padding_idx, left_pad)
    m.weight.data.normal_(0, 0.1)
    return m


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def get_archs():
    return [
        'BN_iwslt_de_en', 'BN_wubi2en', 'BN_en2wubi',
    ]


def _check_arch(args):
    """Check that the specified architecture is valid and not ambiguous."""
    if args.arch not in get_archs():
        raise ValueError('Unknown BN model architecture: {}'.format(
            args.arch))
    if args.arch != 'BN':
        # check that architecture is not ambiguous
        for a in [
                'encoder_embed_dim', 'encoder_layers',
                'decoder_embed_dim', 'decoder_layers',
                'decoder_out_embed_dim']:
            if hasattr(args, a):
                raise ValueError(
                    '--{} cannot be combined with --arch={}'.format(
                        a, args.arch))


def parse_arch(args):
    _check_arch(args)

    if args.arch == 'BN_iwslt_de_en':
        args.encoder_embed_dim = 256
        args.encoder_layers = '(([1, 2, 4, 8, 16], 200, 3, False),)*3'
        args.decoder_embed_dim = 256
        args.decoder_layers = '(([1, 2, 4, 8, 16], 200, 3, True),)*3'
        args.decoder_out_embed_dim = 256
        #  args.decoder_out_embed_dim = 256
    elif args.arch == 'BN_wubi2en':
        args.encoder_embed_dim = 512
        args.encoder_layers = '(([1, 2, 4, 8, 16], 512, 3, False),)*3'
        args.decoder_embed_dim = 512
        args.decoder_layers = '(([1, 2, 4, 8, 16], 512, 3, True),)*3'
        args.decoder_out_embed_dim = 512
    elif args.arch == 'BN_en2wubi':
        args.encoder_embed_dim = 512
        args.encoder_layers = '(([1, 2, 4, 8, 16], 512, 3, False),)*3'
        args.decoder_embed_dim = 512
        args.decoder_layers = '(([1, 2, 4, 8, 16], 512, 3, True),)*3'
        args.decoder_out_embed_dim = 512
    else:
        assert args.arch == 'BN'

    # default architecture
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers',
                                  '(([1, 2, 4, 8, 16], 512, 3, False),)*3')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers',
                                  '(([1, 2, 4, 8, 16], 512, 3, True),)*3')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed',
                                            False)
    return args


def build_model(args, src_dict, dst_dict):
    encoder = BNEncoder(
        src_dict,
        embed_dim=args.encoder_embed_dim,
        convolutions=eval(args.encoder_layers),
        dropout=args.dropout,
        max_positions=args.max_source_positions,
    )
    decoder = BNDecoder(
        dst_dict,
        embed_dim=args.decoder_embed_dim,
        convolutions=eval(args.decoder_layers),
        out_embed_dim=args.decoder_out_embed_dim,
        attention=eval(args.decoder_attention),
        dropout=args.dropout,
        max_positions=args.max_target_positions,
        share_embed=args.share_input_output_embed
    )
    return BNModel(encoder, decoder)
