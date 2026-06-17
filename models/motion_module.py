# This file is originally from AnimateDiff/animatediff/models/motion_module.py at main · guoyww/AnimateDiff
# SPDX-License-Identifier: Apache-2.0 license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file was released under [ Apache-2.0 license], with the full license text available at [https://github.com/guoyww/AnimateDiff?tab=Apache-2.0-1-ov-file#readme].
import torch
import torch.nn.functional as F
from torch import nn

from .attention import CrossAttention, FeedForward, apply_rotary_emb, precompute_freqs_cis

from einops import rearrange, repeat
import math

try:
    import xformers
    import xformers.ops

    XFORMERS_AVAILABLE = True
except ImportError:
    print("xFormers not available")
    XFORMERS_AVAILABLE = False


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


class TemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads                = 8,
        num_transformer_block              = 2,
        num_attention_blocks               = 2,
        norm_num_groups                    = 32,
        temporal_max_len                   = 32,
        zero_initialize                    = True,
        pos_embedding_type                 = "ape",
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads,
            num_layers=num_transformer_block,
            num_attention_blocks=num_attention_blocks,
            norm_num_groups=norm_num_groups,
            temporal_max_len=temporal_max_len,
            pos_embedding_type=pos_embedding_type,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, encoder_hidden_states, attention_mask=None, cached_hidden_state_list=None, position=None):
        hidden_states = input_tensor
        hidden_states, output_hidden_state_list = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask, cached_hidden_state_list, position=position)

        output = hidden_states
        return output, output_hidden_state_list  # list of hidden states


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        num_attention_blocks               = 2,
        norm_num_groups                    = 32,
        temporal_max_len                   = 32,
        pos_embedding_type                 = "ape",
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    num_attention_blocks=num_attention_blocks,
                    temporal_max_len=temporal_max_len,
                    pos_embedding_type=pos_embedding_type,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, cached_hidden_state_list=None, position = None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        output_hidden_state_list = []
        # print(hidden_states.shape)
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        # print(hidden_states.shape)

        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        # B, C, H, W --> B, (H x W), C
        # print('entering TemporalTransformer3DModel', hidden_states.shape)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim).contiguous()
        hidden_states = self.proj_in(hidden_states)
        # print('proj_in B, C, H, W --> B, (H x W), C', hidden_states.shape)
        # Transformer Blocks
        if cached_hidden_state_list is not None:
            n = len(cached_hidden_state_list) // len(self.transformer_blocks)
        else:
            n = 0
        for i, block in enumerate(self.transformer_blocks):
            hidden_states, hidden_state_list = block(hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length, attention_mask=attention_mask,
                                                     cached_hidden_state_list=cached_hidden_state_list[i*n:(i+1)*n] if n else None, position = position)
            # print('out block', i, hidden_states.shape)
            output_hidden_state_list.extend(hidden_state_list)

        # output
        hidden_states = self.proj_out(hidden_states)
        # print('proj_out', hidden_states.shape)
        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output, output_hidden_state_list


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        num_attention_blocks               = 2,
        temporal_max_len                   = 32,
        pos_embedding_type                 = "ape",
    ):
        super().__init__()

        if pos_embedding_type == "gla3":
            dim_tempo = dim+19
        elif "gla" in pos_embedding_type:
            dim_tempo = dim+3
            # print('here !')
        else:
            dim_tempo = dim
        self.attention_blocks = nn.ModuleList(
            [
                TemporalAttention(
                        query_dim=dim_tempo,
                        heads=num_attention_heads,
                        dim_head=attention_head_dim,
                        temporal_max_len=temporal_max_len,
                        pos_embedding_type=pos_embedding_type,
                        query_dim_out = dim
                )
                for i in range(num_attention_blocks)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(dim)
                for i in range(num_attention_blocks)
            ]
        )

        self.ff = FeedForward(dim, dropout=0.0, activation_fn="geglu")
        self.ff_norm = nn.LayerNorm(dim)


    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, cached_hidden_state_list=None, position = None):
        output_hidden_state_list = []
        for i, (attention_block, norm) in enumerate(zip(self.attention_blocks, self.norms)):
            norm_hidden_states = norm(hidden_states)
            residual_hidden_states, output_hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                attention_mask=attention_mask,
                cached_hidden_states=cached_hidden_state_list[i] if cached_hidden_state_list is not None else None,
                position = position
            )
            # print( residual_hidden_states.shape, hidden_states.shape)
            hidden_states = residual_hidden_states + hidden_states
            output_hidden_state_list.append(output_hidden_states)

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output, output_hidden_state_list



class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        dropout = 0.,
        max_len = 32
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.dtype)
        return self.dropout(x)
        
class TemporalPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        dropout = 0.
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model-3
        
    def forward(self, x, position):
        position = position.unsqueeze(1) # T --> T,1
        pe = torch.zeros(1, position.shape[0], self.d_model)
        
        pe[0, :, 0::2] = torch.sin(position/365 * 2 * torch.pi)
        pe[0, :, 1::2] = torch.cos(position/365 * 2 * torch.pi)

        # print(position.shape, x.shape, pe.shape)
        x = x + pe[:, :x.size(1)].to(x.dtype).to(x.device)
        # print('temp encod', x.shape)
        return self.dropout(x)


class MultiTemporalPositionalEncoding(nn.Module):
    def __init__(self, periods=None, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if periods is None:
            periods = [3, 7, 15, 30, 90, 180, 365, 3650]

        self.periods = torch.tensor(periods)

    def forward(self, x, days_position):
        """
        x: (B*D, F, C)
        days_from_solstice: (F,)
        """
        # print(days_position.min(), days_position.max())
        device = x.device
        days_position = days_position.float().unsqueeze(-1)  # (F, 1)
        angles = days_position / self.periods.view(1, -1).to(device)  * 2 * math.pi # (F, len(periods))

        # print(angles.shape) f, len(perdiods)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        days_position_periods = torch.cat([cos, sin], dim=-1)  # (F, 2*len(periods))

        days_position_periods = days_position_periods.unsqueeze(0).expand(x.shape[0], -1, -1).to( x.device )
        # -> (B*D, F, 2*len(periods))

        x = torch.cat((x, days_position_periods), dim=2) #  (B*D, F, C) -->  (B*D, F, C + 2*len(periods))

        return self.dropout(x)

class TimeDeltaEncoding(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        cos_sin = False
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.cos_sin = cos_sin
    def forward(self, x, timedelta_from_lidar, d):
        # x : (b.d),f,c
        # b : batch
        # d : patch h x w
        # f : n frame
        # print(timedelta_from_lidar.min(), timedelta_from_lidar.max())

        d_ = int( math.sqrt( d ) )
        timedelta_from_lidar = F.interpolate( timedelta_from_lidar, (d_, d_))

        # inspired from PositionalEncoding
        if self.cos_sin:
            timedelta_from_lidar = timedelta_from_lidar / 365 * 2 * math.pi
            timedelta_from_lidar[..., 0::2] = torch.sin(timedelta_from_lidar)[..., 0::2]
            timedelta_from_lidar[..., 1::2] = torch.cos(timedelta_from_lidar)[..., 0::2]

        timedelta_from_lidar = timedelta_from_lidar.reshape(timedelta_from_lidar.shape[0], d, 1).contiguous() # nearest, better mean or median ?
        timedelta_from_lidar = rearrange(timedelta_from_lidar, "(b f) d c -> (b d) f c", f=x.shape[1]).to( x.device )

        x = torch.cat( (x, timedelta_from_lidar), dim=2) # (b d) f c+1

        return x
        

class SeasonalEncoding(nn.Module):
    def __init__(self, period=365.0, dropout=0.0):
        super().__init__()
        self.period = period
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, days_from_solstice):
        """
        x: (B*D, F, C)
        days_from_solstice: (F,)
        """
        # print(days_from_solstice.min(), days_from_solstice.max())

        device = x.device
        days = days_from_solstice.to(device).float()  # (F,)

        angle = 2 * math.pi * days / self.period

        cos = torch.cos(angle)
        sin = torch.sin(angle)

        seasonal = torch.stack([cos, sin], dim=-1)  # (F, 2)

        # Expand to match batch*patch dimension
        seasonal = seasonal.unsqueeze(0).expand(x.shape[0], -1, -1)
        # -> (B*D, F, 2)

        x = torch.cat((x, seasonal), dim=2)

        return self.dropout(x)


class TemporalAttention(CrossAttention):
    def __init__(
            self,
            temporal_max_len                   = 32,
            pos_embedding_type                 = "ape",
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.pos_embedding_type = pos_embedding_type
        self._use_memory_efficient_attention_xformers = True

        self.pos_encoder = None
        self.freqs_cis = None
        self.timedelta_encoder = None
        self.temporal_encoder = None
        if self.pos_embedding_type == "ape":
            self.pos_encoder = PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.,
                max_len=temporal_max_len
            )

        elif self.pos_embedding_type == "rope":
            self.freqs_cis = precompute_freqs_cis(
                kwargs["query_dim"],
                temporal_max_len
            )
        elif self.pos_embedding_type == "gla":
            # self.pos_encoder_tempo = TemporalPositionalEncoding( kwargs['query_dim'], dropout=0. )
            self.timedelta_encoder = TimeDeltaEncoding()
            self.seasonal_encoder = SeasonalEncoding()
        elif self.pos_embedding_type == "gla2":
            # self.pos_encoder_tempo = TemporalPositionalEncoding( kwargs['query_dim'], dropout=0. )
            self.temporal_encoder = TemporalPositionalEncoding( kwargs['query_dim'] )
            self.timedelta_encoder = TimeDeltaEncoding(cos_sin=True)
            self.seasonal_encoder = SeasonalEncoding()
        elif self.pos_embedding_type == "gla3":
            self.temporal_encoder = MultiTemporalPositionalEncoding()
            self.timedelta_encoder = TimeDeltaEncoding(cos_sin=True)
            self.seasonal_encoder = SeasonalEncoding()
            
        else:
            raise NotImplementedError

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, cached_hidden_states=None, position=None):
        # TODO: support cache for these
        assert encoder_hidden_states is None
        assert attention_mask is None
        # print('>> attention block', hidden_states.shape)
        d = hidden_states.shape[1]
        d_in = 0
        if cached_hidden_states is None:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            input_hidden_states = hidden_states  # (bxd) f c
        else:
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=1)
            input_hidden_states = hidden_states
            d_in = cached_hidden_states.shape[1]
            hidden_states = torch.cat([cached_hidden_states, hidden_states], dim=1)

        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)

        if self.timedelta_encoder is not None:
            # hidden_states = self.pos_encoder_tempo(hidden_states, position)
            if self.temporal_encoder is not None:
                # print( hidden_states.shape )
                hidden_states = self.temporal_encoder(hidden_states, position[2])
            hidden_states = self.timedelta_encoder(hidden_states, position[0],d)
            hidden_states = self.seasonal_encoder(hidden_states, position[1])
        
        # print('pos encoder', hidden_states.shape)
        encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states[:, d_in:, ...])
        dim = query.shape[-1]

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        
        # print('to_q', query.shape)
        # print('to_k', key.shape)
        # print('to_v', value.shape)
        if self.freqs_cis is not None:
            seq_len = query.shape[1]
            freqs_cis = self.freqs_cis[:seq_len].to(query.device)
            query, key = apply_rotary_emb(query, key, freqs_cis)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)


        use_memory_efficient = XFORMERS_AVAILABLE and self._use_memory_efficient_attention_xformers
        if use_memory_efficient and (dim // self.heads) % 8 != 0:
            # print('Warning: the dim {} cannot be divided by 8. Fall into normal attention'.format(dim // self.heads))
            use_memory_efficient = False

        # attention, what we cannot get enough of
        if use_memory_efficient:
            query = self.reshape_heads_to_4d(query)
            key = self.reshape_heads_to_4d(key)
            value = self.reshape_heads_to_4d(value)

            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            # print('reshape query', query.shape)
            # print('reshape key' , key.shape)
            # print('reshape value', value.shape)
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
                # print('attention', hidden_states.shape)
            else:
                raise NotImplementedError
                # hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
        # print( hidden_states.shape, input_hidden_states.shape)
        return hidden_states, input_hidden_states