# -*- coding: UTF-8 -*-

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.BaseModel import BaseModel
from utils import utils
from einops import rearrange, repeat, einsum


class JKT(BaseModel):
    extra_log_args = ['num_layer', 'd_state','dt_rank','d_conv','d_inner','conv_bias','bias','time_log','win']

    @staticmethod
    def parse_model_args(parser, model_name='JKT'):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layer', type=int, default=1,
                            help='Self-attention layers.')
        parser.add_argument('--d_state', type=int, default=8,
                            help='Self-attention heads.')
        parser.add_argument('--dt_rank', type=int, default=8,
                            help='Self-attention heads.')
        parser.add_argument('--d_conv', type=int, default=4,
                            help='Self-attention heads.')
        parser.add_argument('--d_inner', type=int, default=128,
                            help='Self-attention heads.')
        parser.add_argument('--conv_bias', type=bool, default=True,
                            help='Self-attention heads.')
        parser.add_argument('--bias', type=bool, default=False,
                            help='Self-attention heads.')
        parser.add_argument('--time_log', type=float, default=np.e,
                            help='Log base of time intervals.')
        parser.add_argument('--win', type=int, default=1,
                            help='Log base of time intervals.')
        parser.add_argument('--d_size', type=int, default=32,
                            help='Log base of time intervals.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        super().__init__(model_path=args.model_path)
    def forward(self, feed_dict):




