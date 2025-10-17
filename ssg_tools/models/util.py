# Code based on code from https://github.com/ShunChengWu/3DSSG/tree/cvpr21
# Copyright (c) 2021, ShunChengWu
# All rights reserved.

from __future__ import annotations
from typing import Literal, Optional, Sequence
import torch.nn as nn

InitType = Literal["normal", "xavier_normal", "kaiming", "orthogonal", "xavier_unifrom"]


class InitWeightsMixin:
    def init_weights(
        self, init_type: InitType = "normal", gain: float = 0.02, bias_value: float = 0.0, target_op: Optional[str] = None
    ) -> None:
        """Initialize a torch.nn.Module's weights.
        Source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        Args:
            init_type: The type of initialization. Defaults to 'normal'.
            gain: The gain of the initialization. Defaults to 0.02.
            bias_value: The bias value. Defaults to 0.0.
            target_op: The target op to initialize as str. Defaults to None.

        Raises:
            NotImplementedError: If the init_type is not supported.
        """

        def init_func(m):
            classname = m.__class__.__name__

            if target_op is not None:
                if classname.find(target_op) == -1:
                    return False

            if hasattr(m, "param_initialized"):
                return

            if getattr(m, "weight", None) is not None:
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == "xavier_unifrom":
                    nn.init.xavier_uniform_(m.weight.data, gain=gain)
                elif init_type == "constant":
                    nn.init.constant_(m.weight.data, gain)
                else:
                    raise NotImplementedError()

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, bias_value)
            m.param_initialized = True

        self._init_apply(init_func)

    def _init_apply(self, fn):
        for m in self.children():
            if hasattr(m, "param_initialized"):
                if m.param_initialized is False:
                    m.init_apply(fn)
            else:
                m.apply(fn)
        fn(self)
        return self


def build_mlp(
    dim_list: Sequence[int],
    op: str = "linear",
    activation: str = "relu",
    batch_norm: bool = False,
    dropout: float = 0.0,
    on_last: bool = True,
) -> nn.Sequential:
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        if op == "linear":
            layers.append(nn.Linear(dim_in, dim_out))
        elif op == "conv1d":
            layers.append(nn.Conv1d(dim_in, dim_out, kernel_size=1, bias=True))
        final_layer = i == len(dim_list) - 2
        if not final_layer or on_last:
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leakyrelu":
                layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)
