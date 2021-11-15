import torch.nn as nn
import torch
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding
from .time_embed import TimeEmbedding

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1, is_logkey=True, is_time=False):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=max_len)
        # self.position = nn.DataParallel(self.position)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        # self.segment = nn.DataParallel(self.segment)
        self.time_embed = TimeEmbedding(embed_size=self.token.embedding_dim)
        # self.time_embed = nn.DataParallel(self.time_embed)
        self.dropout = nn.Dropout(p=dropout)
        # self.dropout = nn.DataParallel(self.dropout)
        self.embed_size = embed_size
        self.is_logkey = is_logkey
        self.is_time = is_time
        self.vocab_size = vocab_size

    def forward(self, sequence, segment_label=None, time_info=None, emb = None):
        # print(self.vocab_size)
        # print(sequence.size())
        x = self.position(sequence)
        # print(x.size())
        # print(self.token(sequence).size())
        # if self.is_logkey:
        y = self.token(sequence)
        # print(sequence.size(), len(emb), emb[0].size(), y.size(), sequence)
        # print(sequence.size(), y.size(), emb.size())
        # print(y[0][1],'\n', sequence[0][1], '\n', emb[0][1])
        y = torch.where(torch.stack([sequence for _ in range(y.size(2))], dim = 2) == 4, y, emb)
        # print("now: ", x)
        x = x + y
        # print(x.size())
        # print("Done\n")
        if segment_label is not None:
            x = x + self.segment(segment_label)
        if self.is_time:
            x = x + self.time_embed(time_info)
        return self.dropout(x)
