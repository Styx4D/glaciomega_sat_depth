# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale_init_value = layer_scale_init_value
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # print('\n ON BLOCK')
        input = x
        x = self.dwconv(x)
        # print(f'dwconv : {x.min(), x.max()}')
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        # print(f'norm : {x.min(), x.max()}')
        x = self.pwconv1(x)
        # print(f'pwconv1 : {x.min(), x.max()}')
        x = self.act(x)
        # print(f'act : {x.min(), x.max()}')
        x = self.pwconv2(x)
        # print(f'pwconv2 : {x.min(), x.max()}')
        if self.gamma is not None:
            x = self.gamma * x
            # print(f'gamma : {x.min(), x.max()}')
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        # print(f'drop_path : {x.min(), x.max()}')
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(normalized_shape))
        self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.init_weights()

    def init_weights(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    r"""
    Code adapted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.pyConvNeXt

    A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        patch_size (int | None): Pseudo patch size. Used to resize feature maps to those of a ViT with a given patch size. If None, no resizing is performed
    """

    def __init__(
        self,
        # original ConvNeXt arguments
        in_chans: int = 3,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        # DINO arguments
        patch_size: int | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        del ignored_kwargs

        # ==== ConvNeXt's original init =====
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # ==== End of ConvNeXt's original init =====

        # ==== DINO adaptation ====
        self.head = nn.Identity()  # remove classification head
        self.embed_dim = dims[-1]
        self.embed_dims = dims  # per layer dimensions
        self.n_blocks = len(self.downsample_layers)  # 4
        self.chunked_blocks = False
        self.n_storage_tokens = 0  # no registers

        self.norms = nn.ModuleList([nn.Identity() for i in range(3)])
        self.norms.append(self.norm)

        self.patch_size = patch_size
        self.input_pad_size = 4  # first convolution with kernel_size = 4, stride = 4

    def init_weights(self):
        self.apply(self._init_weights)
        for stage_id, stage in enumerate(self.stages):
            for block_id, block in enumerate(stage):
                if block.gamma is not None:
                    nn.init.constant_(self.stages[stage_id][block_id].gamma, block.layer_scale_init_value)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        if isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        output = []
        for x, masks in zip(x_list, masks_list):
            h, w = x.shape[-2:]

            # print(x.shape)
            for i in range(4):
                x = self.downsample_layers[i](x)
                # print('down', i, x.shape)
                x = self.stages[i](x)
                # print('stages', i, x.shape)
            # print(x.shape)
            x_pool = x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
            x = torch.flatten(x, 2).transpose(1, 2)

            # concat [CLS] and patch tokens as (N, HW + 1, C), then normalize
            x_norm = self.norm(torch.cat([x_pool.unsqueeze(1), x], dim=1))
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_storage_tokens": x_norm[:, 1 : self.n_storage_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )

        return output

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

    def _get_intermediate_layers(self, x, n=1):
        h, w = x.shape[-2:]
        output, total_block_len = [], len(self.downsample_layers)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i in range(total_block_len):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in blocks_to_take:
                x_pool = x.mean([-2, -1])
                x_patches = x
                if self.patch_size is not None:
                    # Resize output feature maps to that of a ViT with given patch_size
                    x_patches = nn.functional.interpolate(
                        x,
                        size=(h // self.patch_size, w // self.patch_size),
                        mode="bilinear",
                        antialias=True,
                    )
                output.append(
                    [
                        x_pool,  # CLS (B x C)
                        x_patches,  # B x C x H x W
                    ]
                )
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def simple_interm_layers(self,x):
        out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out.append(x)
        return out
        
    def get_intermediate_layers(
        self,
        x,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        outputs = self._get_intermediate_layers(x, n)

        if norm:
            nchw_shapes = [out[-1].shape for out in outputs]
            if isinstance(n, int):
                norms = self.norms[-n:]
            else:
                norms = [self.norms[i] for i in n]
            outputs = [
                (
                    norm(cls_token),  # N x C
                    norm(patches.flatten(-2, -1).permute(0, 2, 1)),  # N x HW x C
                )
                for (cls_token, patches), norm in zip(outputs, norms)
            ]
            if reshape:
                outputs = [
                    (cls_token, patches.permute(0, 2, 1).reshape(*nchw).contiguous())
                    for (cls_token, patches), nchw in zip(outputs, nchw_shapes)
                ]
        elif not reshape:
            # force B x N x C format for patch tokens
            outputs = [(cls_token, patches.flatten(-2, -1).permute(0, 2, 1)) for (cls_token, patches) in outputs]
        class_tokens = [out[0] for out in outputs]
        outputs = [out[1] for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)


convnext_sizes = {
    "tiny": dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
    ),
    "small": dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
    ),
    "base": dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ),
    "large": dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
    ),
}


def get_convnext_arch(arch_name):
    size_dict = None
    query_sizename = arch_name.split("_")[1]
    try:
        size_dict = convnext_sizes[query_sizename]
    except KeyError:
        raise NotImplementedError("didn't recognize vit size string")

    return partial(
        ConvNeXt,
        **size_dict,
    )

###### Convnext + DPT
# code adapted from convnext + Dav2 git

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        
        output = self.out_conv(output)

        return output
    

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        out_chan = 2
    ):
        super(DPTHead, self).__init__()
        
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for in_channel,out_channel in zip(in_channels,out_channels)
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(True),
            # nn.Tanh(),
            nn.Conv2d(head_features_2, out_chan, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True),
            # nn.Identity(),
        )
    
    def forward(self, out_features, patch_size = 14):
        out = []
        for i, x  in enumerate(out_features):
            # B, patch_h*patch_w, embed --> B, embed, patch_h, patch_w
            # print(x.shape)
            # x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            # print(x.shape)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
            # print(x.shape)

        # shapes :  B, 256, H, W
        #           B, 512, H//4, W//4
        #           B, 1024, H//16, W//16
        #           B, 1024, H//64, W//64
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # print( 'layer_1_rn : ', layer_1_rn.shape)
        # print( 'layer_2_rn : ', layer_2_rn.shape)
        # print( 'layer_3_rn : ', layer_3_rn.shape)
        # print( 'layer_4_rn : ', layer_4_rn.shape)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn, size = layer_1_rn.shape[2:])
        
        # print( 'path_4 : ', path_4.shape)
        # print( 'path_3 : ', path_3.shape)
        # print( 'path_2 : ', path_2.shape)
        # print( 'path_1 : ', path_1.shape)
        

        out = self.scratch.output_conv1(path_1)
        # print(out.shape)
        # out = F.interpolate(out, (int(patch_h * patch_size), int(patch_w * patch_size)), mode="bilinear", align_corners=True)
        # print(out.shape)
        out = self.scratch.output_conv2(out)
        # print(out.shape)
        return out

class ConvNeXtDPT(nn.Module):
    def __init__(
        self,
        in_chans=3,
        out_chans=2,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        drop_path_rate=0.,
        layer_scale_init_value=1e-6,
        pretrain = None
    ):
        super().__init__()

        self.convnext = ConvNeXt( in_chans=in_chans, depths=depths, dims=dims, drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value)

        if pretrain:
            encoder_state_dict = torch.load(pretrain, weights_only = True)
            # print(encoder_state_dict.keys())
            if in_chans != 3:
                enc_w = encoder_state_dict['downsample_layers.0.0.weight']
                encoder_state_dict['downsample_layers.0.0.weight'] = torch.ones( (enc_w.shape[0], in_chans, enc_w.shape[2], enc_w.shape[3]))
                encoder_state_dict['downsample_layers.0.0.bias'] = torch.zeros( enc_w.shape[0] )

            incompatible_keys = self.convnext.load_state_dict(encoder_state_dict)
            # if str(incompatible_keys) != '<All keys matched successfully>':
            #     logger.warning(incompatible_keys)

        self.dpt = DPTHead( in_channels=dims, out_chan=out_chans)

    def forward(self, x, freeze_bbone = True):
        B,C,H,W = x.shape
        # [torch.Size([32, 128, 64, 64]), torch.Size([32, 256, 32, 32]), torch.Size([32, 512, 16, 16]), torch.Size([32, 1024, 8, 8])]
        # feat de la taille [B, dim, **(H,W//i) ] for i in (4,8,16,32) for dim in dims
        # feats = self.convnext.get_intermediate_layers( x, 4, reshape = True )
        if freeze_bbone:
            with torch.no_grad():
                feats = self.convnext.simple_interm_layers( x )
        else:
            feats = self.convnext.simple_interm_layers( x )
        # print( [f.shape for f in feats])

        out = self.dpt( feats )

        return out

    def get_convnext_features( self, x, freeze = True):
        if freeze:
            with torch.no_grad():
                feats = self.convnext.simple_interm_layers( x )
        else:
            feats = self.convnext.simple_interm_layers( x )

        return feats

    def infer_dpt( self, feats, freeze = True):
        if freeze:
            with torch.no_grad():
                out = self.dpt( feats )
        else:
            out = self.dpt( feats )

        return out

# model = ConvNeXtDPT(in_chans=7).cuda()

# b_size = 2
# H, W = 512, 512

# test = torch.randn((b_size, 7, H, W)).float().cuda()

# with torch.no_grad():
#     out = model( test )
# print(out.shape)

""" Temporal DPT """
from .motion_module import TemporalModule

# adapted from
# source : https://github.com/DepthAnything/Video-Depth-Anything/blob/main/video_depth_anything/dpt_temporal.py

from easydict import EasyDict


class DPTHeadTemporal(nn.Module):
    def __init__(self, 
        in_channels=[128, 256, 512, 1024], 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False,
        num_frames=32,
        pe='gla',
        # pe = 'ape',
        out_chan = 2
    ):
        super(DPTHeadTemporal, self).__init__()

        assert num_frames > 0
        motion_module_kwargs = EasyDict(num_attention_heads                = 8,
                                        num_transformer_block              = 1,
                                        num_attention_blocks               = 2,
                                        temporal_max_len                   = num_frames,
                                        zero_initialize                    = True,
                                        pos_embedding_type                 = pe)

        self.motion_modules = nn.ModuleList([
            TemporalModule(in_channels=out_channels[2], 
                           **motion_module_kwargs),
            TemporalModule(in_channels=out_channels[3],
                           **motion_module_kwargs),
            TemporalModule(in_channels=features,
                           **motion_module_kwargs),
            TemporalModule(in_channels=features,
                           **motion_module_kwargs)
        ])

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for in_channel,out_channel in zip(in_channels,out_channels)
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(True),
            # nn.Tanh(),
            nn.Conv2d(head_features_2, out_chan, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True),
            # nn.Identity(),
        )

    def forward(self, out_features, frame_length, micro_batch_size=4, cached_hidden_state_list=None, position=None):
        out = []
        for i, x  in enumerate(out_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)

        # shapes :  B, 256, H, W
        #           B, 512, H//4, W//4
        #           B, 1024, H//16, W//16
        #           B, 1024, H//64, W//64

        layer_1, layer_2, layer_3, layer_4 = out

        B, T = layer_1.shape[0] // frame_length, frame_length
        if cached_hidden_state_list is not None:
            N = len(cached_hidden_state_list) // len(self.motion_modules)
        else:
            N = 0
        # print('layer_3',layer_3.shape)
        layer_3, h0 = self.motion_modules[0](layer_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None, cached_hidden_state_list[0:N] if N else None, position = position)
        layer_3 = layer_3.permute(0, 2, 1, 3, 4).flatten(0, 1)
        # print('layer_3',layer_3.shape)
        # print('---')
        layer_4, h1 = self.motion_modules[1](layer_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None, cached_hidden_state_list[N:2*N] if N else None, position = position)
        layer_4 = layer_4.permute(0, 2, 1, 3, 4).flatten(0, 1)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_4, h2 = self.motion_modules[2](path_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None, cached_hidden_state_list[2*N:3*N] if N else None, position = position)
        path_4 = path_4.permute(0, 2, 1, 3, 4).flatten(0, 1)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_3, h3 = self.motion_modules[3](path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None, cached_hidden_state_list[3*N:] if N else None, position = position)
        path_3 = path_3.permute(0, 2, 1, 3, 4).flatten(0, 1)

        batch_size = layer_1_rn.shape[0]
        if batch_size <= micro_batch_size or batch_size % micro_batch_size != 0:
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn, size = layer_1_rn.shape[2:])

            out = self.scratch.output_conv1(path_1)
            # out = F.interpolate(
            #     out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True
            # )
            ori_type = out.dtype
            with torch.autocast(device_type="cuda", enabled=False):
                out = self.scratch.output_conv2(out.float())

            output = out.to(ori_type) 
        else:
            ret = []
            for i in range(0, batch_size, micro_batch_size):
                path_2 = self.scratch.refinenet2(path_3[i:i + micro_batch_size], layer_2_rn[i:i + micro_batch_size], size=layer_1_rn[i:i + micro_batch_size].shape[2:])
                path_1 = self.scratch.refinenet1(path_2, layer_1_rn[i:i + micro_batch_size], size=layer_1_rn[i:i + micro_batch_size].shape[2:])
                out = self.scratch.output_conv1(path_1)
                # out = F.interpolate(
                #     out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True
                # )
                ori_type = out.dtype
                with torch.autocast(device_type="cuda", enabled=False):
                    out = self.scratch.output_conv2(out.float())
                ret.append(out.to(ori_type))
            output = torch.cat(ret, dim=0)
        
        return output, h0 + h1 + h2 + h3

def load_model( model_type, encoder_type, pretrain_ConvNeXtDPT, pretrain_DPTHeadTemporal, use_weight_from_DPT, freeze_bbone, device, in_chans = 7 ):
    model_ConvNeXtDPT = ConvNeXtDPT( in_chans = 7 )

    if pretrain_ConvNeXtDPT is not None:
        model_ConvNeXtDPT.load_state_dict(torch.load(pretrain_ConvNeXtDPT, weights_only=True))
    
    if freeze_bbone:
        model_ConvNeXtDPT = model_ConvNeXtDPT.eval().to( device )
    else:
        model_ConvNeXtDPT = model_ConvNeXtDPT.train().to( device )

    if model_type == 'multi':
        temporal_head = DPTHeadTemporal(pe=encoder_type)

        if pretrain_DPTHeadTemporal is not None:
            temporal_head.load_state_dict(torch.load(pretrain_DPTHeadTemporal, weights_only=True))
        elif use_weight_from_DPT:
            temporal_head.load_state_dict( model_ConvNeXtDPT.dpt.state_dict(), strict=False)
        
        temporal_head = temporal_head.train().to(device)
    else:
        temporal_head = None
    
    return model_ConvNeXtDPT, temporal_head