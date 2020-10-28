import paddle
import paddle.nn as nn
import paddle.nn.functional as F

paddle.disable_static()
__author__ = "Xian-Yang Qi"

class ScaledDotProductAttention(nn.Layer):
    "Scaled Dot-Product Attention"
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        "please use model.eval() to predict when attn_dropout is not 0"
        self.dropout = nn.Dropout(p=attn_dropout)
    def forward(self, q, k, v, mask=None):

        attn = paddle.matmul(q/self.temperature, k, transpose_y=True)

        "mask_select in paddle is not the same as mask_filled"
        "so when I finish other parts of transformer, i need to repair the mask"
        if mask is not None:
            attn = attn * mask
        attn = self.dropout(F.softmax(attn, axis=-1))

        output = paddle.matmul(attn, v)

        return output, attn

        
