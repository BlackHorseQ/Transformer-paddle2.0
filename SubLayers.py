import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = 'XianYang Qi'
paddle.disable_static()

class MultiHeadAttention(nn.Layer):
    "Multi-Head attention module"

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias_attr=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias_attr=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias_attr=False)
        self.fc   = nn.Linear(n_head * d_v, d_model, bias_attr=False)

        self.attention = ScaledDotProductAttention(temperature= d_k**0.5)

        self.dropout   = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model, epsilon=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        "sz_b: batch_size"
        sz_b, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]

        residual = q
        
        q = self.w_qs(q).reshape((sz_b, len_q, n_head, d_k))
        k = self.w_ks(k).reshape((sz_b, len_k, n_head, d_k))
        v = self.w_vs(v).reshape((sz_b, len_v, n_head, d_v))

        
        q, k, v = q.transpose([0, 2, 1, 3]), k.transpose([0, 2, 1, 3]), v.transpose([0, 2, 1, 3])

        if mask is not None:
            mask = mask.unsqueeze(1)
        
        q, attn = self.attention(q, k, v, mask=mask)
        

        q = q.transpose([0, 2, 1, 3]).reshape((sz_b, len_q, -1))
        q = self.dropout(self.fc(q))

        q += residual
        q = self.layer_norm(q)

        return q, attn
class PositionwiseForward(nn.Layer):
    "A two-feed-forward-layer module"
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, epsilon=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x