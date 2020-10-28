import paddle.nn as nn
import paddle
from transformer.SubLayers import MultiHeadAttention, PositionwiseForward

__author__ = 'XianYang Qi'

paddle.disable_static()

class EncoderLayer(nn.Layer):
    "Compose with 2 layers"
    "first layer is MultiHeadAttention"
    "second layer is PositionwiseForward"
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        "d_model: dimension of model, eg: input is NCHW, W is the same as d_model"
        "d_inner: dimension of model"
        super(EncoderLayer, self).__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn  = PositionwiseForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, self_attn_mask=None):

        enc_output, enc_self_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=self_attn_mask)
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_self_attn


class DecoderLayer(nn.Layer):
    "Compose with 3 layers"
    "Fisrt and Second layer is MultiheadAttetion"
    "Third is PositionwiseForward"
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn  = PositionwiseForward(d_model, d_inner, dropout=dropout)
    def forward(self, dec_input, enc_output, self_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=self_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn