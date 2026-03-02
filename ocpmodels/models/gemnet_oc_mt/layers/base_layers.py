"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import Callable

import torch
import torch.nn as nn

from ..config import BackboneConfig
from ..initializers import he_orthogonal_init


class MoEDense(nn.Module):
    """
    Combines dense layer with scaling for silu activation.

    Arguments
    ---------
    in_features: int
        Input embedding size.
    out_features: int
        Output embedding size.
    bias: bool
        True if use bias.
    activation: str
        Name of the activation function to use.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        activation=None,
        num_modules=8,
        sparse=True,
        top_k=2,
        ln: bool | str | None = None,
        dropout: float | None = None,
        scale_dim: bool = False,
    ):
        super().__init__()

        self.scale_dim = scale_dim

        self.router = nn.Sequential(
            nn.Linear(in_features, num_modules, bias=bias),
            nn.Softmax(dim=-1),
        )

        self.modules = nn.ModuleList(
            [
                nn.Linear(in_features, out_features, bias=bias)
                for _ in range(num_modules)
            ]
        )

        self.top_k = top_k
        self.sparse = sparse

        self.reset_parameters()

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["silu", "swish"]:
            self._activation = ScaledSiLU()
        elif activation is None:
            self._activation = nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet)."
            )

        if ln is None:
            ln = BackboneConfig.instance().ln

        if dropout is None:
            dropout = BackboneConfig.instance().dropout

        if isinstance(self._activation, nn.Identity):
            ln = False
            # dropout = None

        self.dropout = (
            nn.Dropout(dropout) if dropout is not None else nn.Identity()
        )

        self.ln_kind = "pre" if isinstance(ln, bool) else ln
        match ln:
            case True | "pre":
                self.ln = nn.LayerNorm(in_features)
            case "post":
                self.ln = nn.LayerNorm(out_features)
            case False:
                self.ln = nn.Identity()
            case _:
                raise ValueError(
                    f"ln must be bool or 'pre' or 'post' but got {ln}"
                )

    def reset_parameters(
        self,
        initializer=he_orthogonal_init,
        linear_bias_initializer: Callable[[nn.Parameter], None] | None = None,
    ):
        # initializer(self.router.weight)
        # if self.linear.bias is not None:
        #     _ = self.linear.bias.data.fill_(0)

        # if ln_initializer is not None and isinstance(self.ln, nn.LayerNorm):
        #     ln_initializer(self.ln.weight)
        #     _ = self.ln.bias.data.fill_(0)
        # Reset parameters for the router
        initializer(self.router[0].weight)
        if (
            linear_bias_initializer is not None
            and self.router[0].bias is not None
        ):
            linear_bias_initializer(self.router[0].bias)

        # Reset parameters for the modules
        for module in self.modules:
            initializer(module.weight)
            if linear_bias_initializer is not None and module.bias is not None:
                linear_bias_initializer(module.bias)

    def forward(self, x):
        if self.ln_kind == "pre":
            x = self.ln(x)

        # output router weights after softmax
        router_weights = self.router(x)

        # choose top_k values from router weights
        top_k_values, top_k_idx = torch.topk(router_weights, self.top_k, dim=1)

        # Normalize the top_k values to make their sum equal to 1: (bsz, top_k) -> (bsz, top_k)
        top_k_values_normalized = top_k_values / top_k_values.sum(
            dim=1, keepdim=True
        )

        # Create a new zero tensor with the same shape as 'routing': (bsz, nnodes)
        new_weights = torch.zeros_like(router_weights)

        # Scatter the normalized top_k values back into the new tensor at the corresponding indices: (bsz, nnodes)
        new_weights.scatter_(1, top_k_idx, top_k_values_normalized)
        router_weights = new_weights

        module_outputs = torch.stack(
            [module(x) for module in self.modules], dim=1
        )
        # print("module_outputs.shape", module_outputs.shape)
        # (bsz, module_embedding_dim)
        output = torch.einsum("ij,ijk->ik", router_weights, module_outputs)

        x = self._activation(output)

        if self.ln_kind == "post":
            x = self.ln(x)

        x = self.dropout(x)

        if self.scale_dim:
            x = x * (self.linear.weight.shape[1] ** -0.5)

        return x


class Dense(nn.Module):
    """
    Combines dense layer with scaling for silu activation.

    Arguments
    ---------
    in_features: int
        Input embedding size.
    out_features: int
        Output embedding size.
    bias: bool
        True if use bias.
    activation: str
        Name of the activation function to use.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        activation=None,
        ln: bool | str | None = None,
        dropout: float | None = None,
        scale_dim: bool = False,
    ):
        super().__init__()

        self.scale_dim = scale_dim

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["scaled_silu", "scaled_swish"]:
            self.activation = ScaledSiLU()
        elif activation in ["silu", "swish"]:
            # self.activation = nn.SiLU()
            self.activation = ScaledSiLU()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet)."
            )

        if ln is None:
            ln = BackboneConfig.instance().ln

        if dropout is None:
            dropout = BackboneConfig.instance().dropout

        if isinstance(self.activation, nn.Identity):
            ln = False
            # dropout = None

        self.dropout = (
            nn.Dropout(dropout) if dropout is not None else nn.Identity()
        )

        self.ln_kind = "pre" if isinstance(ln, bool) else ln
        match ln:
            case True | "pre":
                self.ln = nn.LayerNorm(in_features)
            case "post":
                self.ln = nn.LayerNorm(out_features)
            case False:
                self.ln = nn.Identity()
            case _:
                raise ValueError(
                    f"ln must be bool or 'pre' or 'post' but got {ln}"
                )

    def reset_parameters(
        self,
        initializer=he_orthogonal_init,
        ln_initializer: Callable[[nn.Parameter], None] | None = None,
    ):
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            _ = self.linear.bias.data.fill_(0)

        if ln_initializer is not None and isinstance(self.ln, nn.LayerNorm):
            ln_initializer(self.ln.weight)
            _ = self.ln.bias.data.fill_(0)

    def forward(self, x):
        if self.ln_kind == "pre":
            x = self.ln(x)
        x = self.linear(x)
        x = self.activation(x)
        if self.ln_kind == "post":
            x = self.ln(x)
        x = self.dropout(x)
        if self.scale_dim:
            x = x * (self.linear.weight.shape[1] ** -0.5)
        return x


class ScaledSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class ResidualLayer(nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Arguments
    ---------
    units: int
        Input and output embedding size.
    nLayers: int
        Number of dense layers.
    layer: nn.Module
        Class for the layers inside the residual block.
    layer_kwargs: str
        Keyword arguments for initializing the layers.
    """

    def __init__(
        self,
        units: int,
        nLayers: int = 2,
        layer=Dense,
        **layer_kwargs,
    ):
        super().__init__()

        self.dense_mlp = nn.Sequential(
            *[
                layer(
                    in_features=units,
                    out_features=units,
                    bias=False,
                    **layer_kwargs,
                )
                for _ in range(nLayers)
            ]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2)

    def forward(self, input):
        x = self.dense_mlp(input)
        x = input + x
        x = x * self.inv_sqrt_2
        return x


class MoEResidualLayer(nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Arguments
    ---------
    units: int
        Input and output embedding size.
    nLayers: int
        Number of dense layers.
    layer: nn.Module
        Class for the layers inside the residual block.
    layer_kwargs: str
        Keyword arguments for initializing the layers.
    """

    def __init__(
        self,
        units: int,
        nLayers: int = 2,
        layer=MoEDense,
        **layer_kwargs,
    ):
        super().__init__()

        self.dense_mlp = nn.Sequential(
            *[
                layer(
                    in_features=units,
                    out_features=units,
                    bias=False,
                    **layer_kwargs,
                )
                for _ in range(nLayers)
            ]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2)

    def forward(self, input):
        x = self.dense_mlp(input)
        x = input + x
        x = x * self.inv_sqrt_2
        return x
