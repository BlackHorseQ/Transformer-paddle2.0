import paddle
import paddle.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = 'Xianyang Qi'

paddle.disable_static()



def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsquent_mask(seq):
    "for masking out the subsequent info."
    sz_b, len_s = seq.shape[0], seq.shape[1]
    "!!torch.ones((1, sz_b, len_s))!!"
    subsequent_mask = (1 - paddle.triu(paddle.ones((1, len_s, len_s)), diagonal=1))

    return subsequent_mask

class PositionalEncoding(nn.Layer):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return paddle.to_tensor(sinusoid_table, dtype='float32').unsqueeze(0)

    def forward(self, x):

        return x + paddle.cast(self.pos_table[:, :x.shape[1]], dtype='float32').detach()
class Encoder(nn.Layer):
    def __init__(self, n_src_vocab=200, d_word_vec=20, n_layers=3, n_head=2, 
        d_k=10, d_v=10, d_model=20, d_inner=10, pad_idx= 0, dropout=0.1, n_position=200, emb_weight=None):
        "args:"
        "n_src_vocab(int): the number of vocabulary of input"
        "src_pad_idx(int): the index of padding word of input"
        "d_word_vec(int) : the dimension of word2vec and d_word_vec is equal to d_model"
        "d_inner(int):     the number of hidden units of PositionwiseForward layer"
        "n_layers(int): the number of decoder layer and encoder layer"
        "n_head(int): the number of attention head"
        "d_k: dimension of d matrix"
        "d_v: dimension of v matrix"
        "src_emb_weight: weight of input w2v"
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, sparse=True, padding_idx=pad_idx)

        if emb_weight is not None:
            self.src_word_emb.weight.set_value(emb_weight)
            self.src_word_emb.stop_gradient=True
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout      = nn.Dropout(dropout)
        self.layer_stack  = nn.LayerList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model, epsilon=1e-6)
    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []
        # print(src_seq.shape)
        "embeding and positional encoding"
        enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq)))
        enc_output = self.layer_norm(enc_output)
        

        "encoder layer"
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, self_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output, 

class Decoder(nn.Layer):
    
    def __init__(self, n_trg_vocab=200, d_word_vec=20, n_layers=2, n_head=2, d_k=10, d_v=10,
        d_model=20, d_inner=10, pad_idx=0, dropout=0.1, n_position=200, emb_weight=None):
        "args:"
        "n_trg_vocab(int): the number of vocabulary of output"
        "trg_pad_idx(int): the index of padding word of output"
        "d_word_vec(int) : the dimension of word2vec and d_word_vec is equal to d_model"
        "d_inner(int):     the number of hidden units of PositionwiseForward layer"
        "n_layers(int): the number of decoder layer and encoder layer"
        "n_head(int): the number of attention head"
        "d_k: dimension of d matrix"
        "d_v: dimension of v matrix"
        "trg_emb_weight: weight of ouput w2v"
        super().__init__()
        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        if emb_weight is not None:
            self.trg_word_emb.weight.set_value(emb_weight)
            self.trg_word_emb.stop_gradient=True
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout      = nn.Dropout(dropout)
        self.layer_stack  = nn.LayerList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, epsilon=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = self.dropout(self.position_enc(self.trg_word_emb(trg_seq)))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, self_attn_mask=trg_mask, dec_enc_attn_mask=src_mask
            )
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output, 

class Transformer(nn.Layer):

    def __init__(
        self, n_src_vocab, n_trg_vocab, src_pad_idx=0, trg_pad_idx=0, 
        d_word_vec=512, d_model=512, d_inner=2048,
        n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
        src_emb_weight=None, trg_emb_weight=None,
        trg_emd_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
    ):  
        "args:"
        "n_src_vocab(int): the number of vocabulary of input"
        "n_trg_vocab(int): the number of vocabulary of output"
        "src_pad_idx(int): the index of padding word of input"
        "trg_pad_idx(int): the index of padding word of output"
        "d_word_vec(int) : the dimension of word2vec and d_word_vec is equal to d_model"
        "d_inner(int):     the number of hidden units of PositionwiseForward layer"
        "n_layers(int): the number of decoder layer and encoder layer"
        "n_head(int): the number of attention head"
        "d_k: dimension of d matrix"
        "d_v: dimension of v matrix"
        "src_emb_weight: weight of input w2v"
        "trg_emb_weight: weight of ouput w2v"

        super(Transformer, self).__init__()
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx


        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, pad_idx=src_pad_idx, d_word_vec=d_word_vec,
            n_layers=n_layers, n_head=n_head, d_model=d_model, d_inner=d_inner,
            d_k=d_k, d_v=d_v, dropout=dropout, n_position=n_position,
            emb_weight=src_emb_weight)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, pad_idx=trg_pad_idx, d_word_vec=d_word_vec,
            n_layers=n_layers, n_head=n_head, d_model=d_model, d_inner=d_inner,
            d_k=d_k, d_v=d_v, dropout=dropout, n_position=n_position,
            emb_weight=trg_emb_weight)
        
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias_attr=False)


        "At present, there is not nn.initializer.Xavier uniforming parameters, I'm wating for the official workers to rely to me"
        # for p in self.parameters():
        #     if p.dim()>1:
        #         nn.initializer.Xavier()

        assert d_model == d_word_vec, 'To facilitate the residual connections, the dimensions of all module outputs shall be the same'
        
        self.x_logit_scale = 1.
        "share the weight of w2v of decoder with final linear layer"
        if trg_emd_prj_weight_sharing:
            weight = self.decoder.trg_word_emb.weight.numpy()
            weight = np.transpose(weight)
            self.trg_word_prj.weight.set_value(weight)
            self.x_logit_scale            = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            weight = self.decoder.trg_word_emb.weight.numpy()
            self.encoder.src_word_emb.weight.set_value(weight)
        
    def forward(self, src_seq, trg_seq):
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)

        # print(get_pad_mask(trg_seq, self.trg_pad_idx).numpy(), get_subsquent_mask(trg_seq).numpy())
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx).numpy().astype(bool) & get_subsquent_mask(trg_seq).numpy().astype(bool)
        trg_mask = paddle.to_tensor(trg_mask)
        # print(trg_mask.shape)
        enc_output, *_ = self.encoder(src_seq, src_mask)

        # print(trg_seq.shape, enc_output.shape)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)

        seq_logit      = self.trg_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.reshape((-1, seq_logit.shape[2]))