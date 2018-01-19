# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from fairseq.data import LanguagePairDataset
from fairseq.modules import BeamableMM
from fairseq.modules import ResBlock

from . import FairseqEncoder, FairseqIncrementalDecoder, FairseqModel


def make_positions(tokens, padding_idx, left_pad, offset=0):
    seqlen = tokens.size(1)
    if not hasattr(make_positions, 'range'):
        make_positions.range = tokens.new()
    if make_positions.range.numel() < offset + seqlen:
        # offset positions by the padding index
        torch.arange(padding_idx + 1, padding_idx + 1 + offset + seqlen,
                     out=make_positions.range)
    mask = tokens.ne(padding_idx)
    positions = make_positions.range[offset:offset+seqlen].expand_as(tokens)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return tokens.clone().masked_scatter_(mask, positions[mask])


class BytenetModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.encoder.num_attention_layers = sum(layer is not None for layer in decoder.attention)


class BytenetEncoder(FairseqEncoder):
    """Convolutional encoder"""
    def __init__(self, dictionary, embed_dim=512, max_positions=1024,
                 convolutions=((512, 3, [1, 2, 4, 8, 16], False),) * 3,
                 dropout=0.1):
        # convolutions = (in_channels, kernel_size, init_dilation,
        # causal, num_res_block)
        super().__init__()
        self.dictionary = dictionary
        self.dropout = dropout

        # map token to embedding
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        #  self.embed_positions = Embedding(max_positions, embed_dim, padding_idx)

        # map embedding dim to in_channels
        in_channels = convolutions[0][0]
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)

        # build up convolution layers
        self.convolutions = nn.ModuleList()
        for sets in convolutions:
            for dilation_rate in sets[2]:
                self.convolutions.append(
                    ResBlock(in_channels=sets[0],
                             kernel_size=sets[1],
                             dilation=dilation_rate,
                             causal=sets[3]))

        # remap from in channels to embedding size
        self.fc2 = Linear(in_channels, embed_dim)

    def forward(self, src_tokens, tgt_tokens):
        # TODO: do we really need this?
        #  positions = Variable(make_positions(src_tokens.data, self.dictionary.pad(),
        #                                      left_pad=LanguagePairDataset.LEFT_PAD_SOURCE))

        # embed tokens and positions
        # TODO: why do I need positions?
        #  x = self.embed_tokens(src_tokens) + self.embed_positions(positions)
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = self.embed_tokens(tgt_tokens)

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # temporal convolutions
        for conv in self.convolutions:
            x = conv(x)

        # B x C x T -> B x T x C
        x = x.transpose(2, 1)

        # project back to size of embedding
        x = self.fc2(x)

        # scale gradients (this only affects backward, not forward)
        #  x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = x + input_embedding

        return x, y

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        # TODO: maybe I need to do something here
        return self.embed_positions.num_embeddings - self.dictionary.pad() - 1


#  class AttentionLayer(nn.Module):
#      def __init__(self, conv_channels, embed_dim, bmm=None):
#          super().__init__()
#          # projects from output of convolution to embedding dimension
#          self.in_projection = Linear(conv_channels, embed_dim)
#          # projects from embedding dimension to convolution size
#          self.out_projection = Linear(embed_dim, conv_channels)
#
#          self.bmm = bmm if bmm is not None else torch.bmm
#
#      def forward(self, x, target_embedding, encoder_out):
#          residual = x
#
#          # attention
#          x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
#          x = self.bmm(x, encoder_out[0])
#
#          # softmax over last dim
#          sz = x.size()
#          x = F.softmax(x.view(sz[0] * sz[1], sz[2]))
#          x = x.view(sz)
#          attn_scores = x
#
#          x = self.bmm(x, encoder_out[1])
#
#          # scale attention output
#          s = encoder_out[1].size(1)
#          x = x * (s * math.sqrt(1.0 / s))
#
#          # project back
#          x = (self.out_projection(x) + residual) * math.sqrt(0.5)
#          return x, attn_scores
#
#      def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
#          """Replace torch.bmm with BeamableMM."""
#          if beamable_mm_beam_size is not None:
#              self.bmm = BeamableMM(beamable_mm_beam_size)


class BytenetDecoder(FairseqIncrementalDecoder):
    """Convolutional decoder"""
    def __init__(self, dictionary, embed_dim=512, out_embed_dim=256,
                 max_positions=1024,
                 convolutions=((512, 3, [1, 2, 4, 8, 16], True),) * 3,
                 attention=False, dropout=0.1):
        super().__init__()
        self.register_buffer('version', torch.Tensor([2]))
        self.dictionary = dictionary
        self.dropout = dropout

        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        #  self.embed_positions = Embedding(max_positions, embed_dim, padding_idx)

        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)

        self.convolutions = nn.ModuleList()

        for sets in convolutions:
            for dilation_rate in sets[2]:
                self.convolutions.append(
                    ResBlock(in_channels=sets[0],
                             kernel_size=sets[1],
                             dilation=dilation_rate,
                             causal=sets[3]))

        self.fc2 = Linear(in_channels, out_embed_dim)
        self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)

    def forward(self, input_tokens, encoder_out):
        if self._is_incremental_eval:
            return self.incremental_forward(input_tokens, encoder_out)
        else:
            return self.batch_forward(input_tokens, encoder_out)

    def batch_forward(self, input_tokens, encoder_out):
        """Forward pass for decoding multiple time steps in batch mode."""
        #  positions = Variable(make_positions(input_tokens.data, self.dictionary.pad(),
        #                                      left_pad=LanguagePairDataset.LEFT_PAD_TARGET))
        return self._forward(input_tokens, encoder_out)

    def incremental_forward(self, input_tokens, encoder_out):
        """Forward pass for one time step."""
        # positions is the same for every token when decoding a single step
        #  positions = Variable(input_tokens.data.new(1, 1).fill_(
        #      self.dictionary.pad() + input_tokens.size(1)))

        # keep only the last token for incremental forward pass
        return self._forward(input_tokens[:, -1:], encoder_out)

    def _forward(self, input_tokens, positions, encoder_out):
        # split and transpose encoder outputs
        encoder_a, encoder_b = self._split_encoder_out(encoder_out)

        # embed tokens and positions
        x = self.embed_tokens(input_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        # TODO: do I need to transpose if not incremental eval?
        x = self._transpose_unless_incremental_eval(x)

        # temporal convolutions
        #  avg_attn_scores = None
        #  num_attn_layers = len(self.attention)
        for conv in self.convolutions:
            x = conv(x)

        # T x B x C -> B x T x C
        # TODO: do I need to transpose if not incremental eval?
        x = self._transpose_unless_incremental_eval(x)

        # project back to size of vocabulary
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x

    def reorder_incremental_state(self, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        super().reorder_incremental_state(new_order)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        # TODO: maybe I need to do something here
        return self.embed_positions.num_embeddings - self.dictionary.pad() - 1

    def upgrade_state_dict(self, state_dict):
        # TODO: Can I remove this?
        if state_dict.get('decoder.version', torch.Tensor([1]))[0] < 2:
            # old models use incorrect weight norm dimension
            for i, conv in enumerate(self.convolutions):
                # reconfigure weight norm
                nn.utils.remove_weight_norm(conv)
                self.convolutions[i] = nn.utils.weight_norm(conv, dim=0)
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
        if self._is_incremental_eval:
            return x
        return x.transpose(1, 2)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
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
        'Bytenet', 'Bytenet_en2wubi', 'Bytenet_wubi2en',
    ]


def _check_arch(args):
    """Check that the specified architecture is valid and not ambiguous."""
    if args.arch not in get_archs():
        raise ValueError('Unknown Bytenet model architecture: {}'.format(args.arch))
    if args.arch != 'Bytenet':
        # check that architecture is not ambiguous
        for a in ['encoder_embed_dim', 'encoder_layers', 'decoder_embed_dim', 'decoder_layers',
                  'decoder_out_embed_dim']:
            if hasattr(args, a):
                raise ValueError('--{} cannot be combined with --arch={}'.format(a, args.arch))


def parse_arch(args):
    _check_arch(args)

    if args.arch == 'Bytenet_en2wubi':
        en_convs = '((512, 3, [1, 2, 4, 8, 16], False),) * 3'
        de_convs = '((1024, 3, [1, 2, 4, 8, 16], True),) * 3'
        # TODO: make encoder conv and decoder conv different
        args.encoder_embed_dim = 768
        args.encoder_layers = en_convs
        args.decoder_embed_dim = 768
        args.decoder_layers = de_convs
        args.decoder_out_embed_dim = 512
    elif args.arch == 'Bytenet_wubi2en':
        en_convs = '((512, 3, [1, 2, 4, 8, 16], False),) * 3'
        de_convs = '((1024, 3, [1, 2, 4, 8, 16], True),) * 3'
        # TODO: make encoder conv and decoder conv different
        args.encoder_embed_dim = 768
        args.encoder_layers = en_convs
        args.decoder_embed_dim = 768
        args.decoder_layers = de_convs
        args.decoder_out_embed_dim = 512
    else:
        assert args.arch == 'Bytenet'

    # default architecture
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    return args


def build_model(args, src_dict, dst_dict):
    encoder = BytenetEncoder(
        src_dict,
        embed_dim=args.encoder_embed_dim,
        convolutions=eval(args.encoder_layers),
        dropout=args.dropout,
        max_positions=args.max_source_positions,
    )
    decoder = BytenetDecoder(
        dst_dict,
        embed_dim=args.decoder_embed_dim,
        convolutions=eval(args.decoder_layers),
        out_embed_dim=args.decoder_out_embed_dim,
        attention=eval(args.decoder_attention),
        dropout=args.dropout,
        max_positions=args.max_target_positions,
    )
    return BytenetModel(encoder, decoder)
