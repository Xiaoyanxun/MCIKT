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
        self.skill_num = int(corpus.n_skills)
        self.question_num = int(corpus.n_problems)
        self.interval_time_num = int(corpus.n_interval)
        self.ans_time_num = int(corpus.n_dwells)
        self.emb_size = args.emb_size
        self.d_size = args.d_size
        self.dropout = args.dropout
        self.time_log = args.time_log
        self.gpu = args.gpu
        self.win = args.win
        self.seq = args.max_step

        self.interval_embeddings = nn.Embedding(self.interval_time_num, self.emb_size)
        self.ans_time_embeddings = nn.Embedding(self.ans_time_num, self.emb_size)
        self.skill_embeddings = nn.Embedding(self.skill_num, self.emb_size)
        self.inter_embeddings = nn.Embedding(self.skill_num * 2, self.emb_size)
        self.difficult_param = nn.Embedding(self.question_num, 1)
        self.difficult = nn.Embedding(self.question_num, self.emb_size)
        self.skill_base = torch.nn.Embedding(self.skill_num, 1)
        self.skill_diff = nn.Embedding(self.skill_num, self.emb_size)
        
        
#         self.loss_function = torch.nn.BCELoss(reduction='sum')
        self.loss_function = torch.nn.BCELoss()
        
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.num_layer)])
        self.norm_f = RMSNorm(self.emb_size)
        self.lm_head = nn.Linear(self.emb_size, self.skill_num, bias=False)
        self.lm_head.weight = self.skill_embeddings.weight

        self.alpha_inter_embeddings = torch.nn.Embedding(self.skill_num *2, self.emb_size)
        self.alpha_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
#         self.beta_inter_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
#         self.beta_skill_embeddings = torch.nn.Embedding(self.skill_num , self.emb_size)
        # 设置全局变量 β
        self.beta = nn.Parameter(torch.tensor(1.0))  # 初始化全局变量 β
        
        self.learning_embed_layer = nn.Linear(2 * self.emb_size, self.emb_size)  # input = exercise + answer time+ answer
        torch.nn.init.xavier_normal_(self.learning_embed_layer.weight)  # follow the original paper
        
        self.leanear = nn.Linear(self.emb_size, 1)
        torch.nn.init.xavier_normal_(self.leanear.weight)
        
        self.learning_y_layer = nn.Linear(4 * self.emb_size, self.emb_size)
        
        self.forget_layer = nn.Linear(self.emb_size, 1)
        torch.nn.init.xavier_normal_(self.forget_layer.weight)
        
        self.tanh = nn.Tanh()
    def forward(self, feed_dict):
        skills = feed_dict['skill_seq']        # [batch_size, real_max_step]
        questions = feed_dict['quest_seq']     # [batch_size, real_max_step]
        labels = feed_dict['label_seq']        # [batch_size, real_max_step]
        times = feed_dict['time_seq']        # [batch_size, seq_len]
        ans_time = feed_dict['dwell_seq']
        interval = feed_dict['interval_seq']
        
        q_diff = self.difficult_param(questions)  
        batch_size, sequence_len = skills.size(0), skills.size(1)
        
        interval_time = self.interval_embeddings(interval)
        ans_time = self.ans_time_embeddings(ans_time)  #[batch_size, seq_len, d_size]
        p = self.difficult(questions)
        mask_labels = labels * (labels > -1).long()
        inters = skills + mask_labels * self.skill_num
        skill_data = self.skill_embeddings(skills)
        skill_diff_data = self.skill_diff(skills)
        skill_data_d = skill_data + q_diff * skill_diff_data
        
        inter_data = self.inter_embeddings(inters)
        answer = labels.contiguous().view(-1, 1)  # (batch_size * sequence) * 1
        answer = answer.repeat(1, self.emb_size)  # (batch_size * sequence) * d_a
        answer = answer.view(batch_size, -1, self.emb_size)  # batch_size * sequence * d_a
        
        learning_emd = self.learning_embed_layer(torch.cat((answer, skill_data), 2))  # [batch_size, sequence, d_k]
        learning_y = self.learning_y_layer(torch.cat((answer, skill_data_d, ans_time, interval_time), 2))
        
        seq = skills.shape[1]


        x = learning_emd[:, :seq - self.win,:]
        y = learning_y[:, :seq - self.win,:]

        
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        pred_vector = self.lm_head(x)     #基础
        
        target_item = skills[:, self.win:]
        prediction_sorted = torch.gather(pred_vector, dim=-1, index=target_item.unsqueeze(dim=-1)).squeeze(dim=-1)  #[bs, seq_len]
        
        learning_rate = self.tanh(self.leanear(y)).squeeze(dim=-1)
        prediction_sorted = prediction_sorted * (learning_rate + 1)/2

        #分别计算了 alpha 和 beta 的源嵌入和目标嵌入
        alpha_src_emb = self.alpha_inter_embeddings(inters)  # [bs, seq_len, emb]
        alpha_target_emb = self.alpha_skill_embeddings(skills)
        alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]

        # 使用全局 β 变量进行缩放
        betas = self.beta

        delta_t = (times[:, :, None] - times[:, None, :]).abs().double()   #它记录了每对时间戳之间的绝对时间差。 # [bs, seq_len, seq_len]
        delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

        seq_len = skills.shape[1]
        # 优化时间效应计算：稀疏化
        valid_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
        for i in range(self.win, seq_len):
            for j in range(self.win - 1):
                valid_mask[:, i, i - (j + 1)] = 1
        mask = (torch.from_numpy(valid_mask) == 0)
        mask = mask.cuda() if self.gpu != '' else mask

        # 稀疏化计算时间效应
        cross_effects = alphas * torch.exp(-betas * delta_t)  # [bs, seq_len, seq_len]
        cross_effects = cross_effects.masked_fill(mask, 0)  # 应用掩码
        sum_t = cross_effects.sum(-2)  # [bs, seq_len]
        
        problem = torch.sigmoid(self.forget_layer(p)).squeeze(dim=-1)
        knowledge = torch.sigmoid(prediction_sorted + sum_t[:,self.win:])
        prediction = torch.sigmoid(prediction_sorted + sum_t[:,self.win:] - problem[:,self.win:])
        out_dict = {'prediction': prediction, 'label': labels[:, self.win:].double(), 'knowledge': knowledge, 'problem': problem[:,self.win:]}
        return out_dict

    def loss(self, feed_dict, outdict):
        prediction = outdict['prediction'].flatten()
        label = outdict['label'].flatten()
        mask = label > -1
        loss = self.loss_function(prediction[mask], label[mask])
        return loss

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        skill_seqs = data['skill_seq'].iloc[batch_start: batch_start + real_batch_size].values
        quest_seqs = data['problem_seq'].iloc[batch_start: batch_start + real_batch_size].values
        label_seqs = data['correct_seq'].iloc[batch_start: batch_start + real_batch_size].values
        time_seqs = data['time_seq'].iloc[batch_start: batch_start + real_batch_size].values
        dwell_seqs = data['dwell_seq'].iloc[batch_start: batch_start + real_batch_size].values
        interval_seqs = data['interval_seq'].iloc[batch_start: batch_start + real_batch_size].values

        feed_dict = {
            'skill_seq': torch.from_numpy(utils.pad_lst(skill_seqs)),            # [batch_size, real_max_step]
            'quest_seq': torch.from_numpy(utils.pad_lst(quest_seqs)),            # [batch_size, real_max_step]
            'label_seq': torch.from_numpy(utils.pad_lst(label_seqs, value=-1)),  # [batch_size, real_max_step]
            'time_seq': torch.from_numpy(utils.pad_lst(time_seqs)),              # [batch_size, seq_len]
            'dwell_seq': torch.from_numpy(utils.pad_lst(dwell_seqs)),              # [batch_size, seq_len]
            'interval_seq': torch.from_numpy(utils.pad_lst(interval_seqs)),              # [batch_size, seq_len]
        }
        return feed_dict
    
class ResidualBlock(nn.Module):    #残差块
    def __init__(self, args):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.emb_size)
        self.dropout1 = nn.Dropout(0.3)
        

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.dropout1(self.mixer(self.norm(x))) + x

        return output
    
class MambaBlock(nn.Module):
    def __init__(self, args):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.emb_size, args.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)  #没懂
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.emb_size, bias=args.bias)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)  

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)   #[bs,sq,h_in]逐元素相乘
        
        
        output = self.out_proj(y)  #[bs,sq,h_moel]

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)  还原A并取负号
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """优化后的选择性扫描：并行化计算"""
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # 离散化参数
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))  # [b, l, d_in, n]
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')  # [b, l, d_in, n]

        # 并行扫描（使用 cumsum 实现）
        x = torch.zeros((b, d_in, n), device=deltaA.device)  # 初始状态
        deltaA_cum = torch.cumprod(deltaA, dim=1)  # [b, l, d_in, n]
        deltaB_u_cum = torch.cumsum(deltaB_u * deltaA_cum, dim=1)  # [b, l, d_in, n]
        x = deltaA_cum[:, -1] * x + deltaB_u_cum[:, -1]  # 最终状态

        # 计算输出
        y = einsum(x, C[:, -1, :], 'b d_in n, b n -> b d_in')  # [b, d_in]
        y = y.unsqueeze(1).repeat(1, l, 1)  # [b, l, d_in]
        y = y + u * D  # [b, l, d_in]

        return y


class RMSNorm(nn.Module):   #归一化
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
            
    



