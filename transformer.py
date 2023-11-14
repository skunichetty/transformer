import torch


class Attention(torch.nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        mask: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        a = q @ k.T
        if mask is not None:
            a += mask
        return self.softmax(a / torch.sqrt(self.output_size)) @ v

    def __repr__(self):
        return "Attention"


class SelfAttention(torch.nn.Module):
    def __init__(
        self, input_size: int = 512, output_size: int = 64, mask: bool = False
    ):
        super(SelfAttention, self).__init__()

        # definition for weights that generate query, key, and value matrices
        self.wquery = torch.nn.Parameter(torch.empty(input_size, output_size))
        self.wkey = torch.nn.Parameter(torch.empty(input_size, output_size))
        self.wvalue = torch.nn.Parameter(torch.empty(input_size, output_size))

        # enable position masking
        self.mask = mask
        self.attn = Attention()
        self.init_weights()

    def init_weights(self):
        weights = (self.wquery, self.wkey, self.wvalue)
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="relu")

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        q, k, v = x @ self.wquery, x @ self.wkey, x @ self.wvalue
        if self.mask:
            return self.attn(q, k, v, torch.triu(q @ k.T) * float("-inf"))
        else:
            return self.attn(q, k, v)

    def __repr__(self):
        return "SelfAttention"


class MultiHeadedSelfAttention(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 64,
        heads: int = 8,
        mask: bool = False,
    ):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.heads = torch.nn.ModuleList(
            [SelfAttention(input_size, hidden_size, mask) for _ in range(heads)]
        )
        self.w0 = torch.nn.Parameter(torch.empty(heads * hidden_size, input_size))
        torch.nn.init.kaiming_normal_(self.w0, nonlinearity="relu")

    def forward(self, x):
        embeddings = tuple(map(lambda a: a(x), self.heads))
        concatenated = torch.cat(embeddings, dim=1).to(x.device)
        return concatenated @ self.w0

    def __repr__(self):
        return f"MultiHeadedSelfAttention(embedding_dim={self.input_size}, hidden_dim={self.hidden_size}, heads={len(self.heads)})"


class EncoderDecoderAttention(torch.nn.Module):
    def __init__(self, input_size: int = 512, output_size: int = 64):
        super(EncoderDecoderAttention, self).__init__()

        self.wquery = torch.nn.Parameter(torch.empty(input_size, output_size))
        self.wkey = torch.nn.Parameter(torch.empty(input_size, output_size))
        self.wvalue = torch.nn.Parameter(torch.empty(input_size, output_size))
        self.init_weights()

        self.attn = Attention()

    def init_weights(self):
        weights = (self.wquery, self.wkey, self.wvalue)
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="relu")

    def forward(
        self, x: torch.FloatTensor, embeddings: torch.FloatTensor
    ) -> torch.FloatTensor:
        q, k, v = x @ self.wquery, embeddings @ self.wkey, embeddings @ self.wvalue
        return self.attn(q, k, v)

    def __repr__(self):
        return "EncoderDecoderAttention"


class MultiHeadedEncoderDecoderAttention(torch.nn.Module):
    def __init__(self, input_size: int = 512, hidden_size: int = 64, heads: int = 8):
        super(MultiHeadedEncoderDecoderAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.heads = torch.nn.ModuleList(
            [EncoderDecoderAttention(input_size, hidden_size) for _ in range(heads)]
        )
        self.w0 = torch.nn.Parameter(torch.empty(heads * hidden_size, input_size))
        torch.nn.init.kaiming_normal_(self.w0, nonlinearity="relu")

    def forward(
        self, x: torch.FloatTensor, hidden: torch.FloatTensor
    ) -> torch.FloatTensor:
        embeddings = tuple(map(lambda a: a(x, hidden), self.heads))
        concatenated = torch.cat(embeddings, dim=1).to(x.device)
        return concatenated @ self.w0

    def __repr__(self):
        return f"MultiHeadedEncoderDecoderAttention(embedding_dim={self.input_size}, hidden_dim={self.hidden_size}, heads={len(self.heads)})"


class TransformerNN(torch.nn.Module):
    def __init__(self, size=512):
        super(TransformerNN, self).__init__()
        self.size = size
        self.linear1 = torch.nn.Linear(size, size * 4, bias=True)
        self.linear2 = torch.nn.Linear(size * 4, size, bias=True)
        self.relu = torch.nn.ReLU()
        self.init_weights()

    def init_weights(self):
        weights = (self.linear1.weight, self.linear2.weight)
        biases = (self.linear1.bias, self.linear2.bias)
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="relu")
        for bias in biases:
            torch.nn.init.constant_(bias, 0.0)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

    def __repr__(self):
        return f"DenseNN(size={self.size})"


class FourierEncoder2d(torch.nn.Module):
    def __init__(self, dropout_probability: float, d_model: int, seq_length: int):
        super(FourierEncoder2d, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout_probability)
        exp = 2 * (torch.div(torch.arange(d_model), 2, rounding_mode="trunc") / d_model)
        denom = 10000**exp
        pos_encodings = torch.arange(seq_length).view(-1, 1) / denom
        pos_encodings[:, 0::2] = torch.sin(pos_encodings[:, 0::2])
        pos_encodings[:, 1::2] = torch.cos(pos_encodings[:, 1::2])
        self.register_buffer("pe", pos_encodings)

    def forward(self, x: torch.FloatTensor):
        N, D = x.shape
        pos_encoding = self._positional_encoding(N, D)
        return self.dropout(x + pos_encoding * x)


class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        seq_length,
        input_size=512,
        hidden_size=64,
        heads=8,
        enable_pos_encoding=False,
    ):
        super(EncoderBlock, self).__init__()
        self.sa = MultiHeadedSelfAttention(input_size, hidden_size, heads)
        self.ln1 = torch.nn.LayerNorm(input_size)
        self.nn = TransformerNN(input_size)
        self.ln2 = torch.nn.LayerNorm(input_size)
        self.dropout = torch.nn.Dropout(p=0.1)

        if enable_pos_encoding:
            self.pos_encoder = FourierEncoder2d(0.4, input_size, seq_length)
        else:
            self.pos_encoder = lambda x: x

    def forward(self, x):
        x = self.pos_encoder(x)
        x += self.dropout(self.sa(x))
        x = self.ln1(x)
        x += self.dropout(self.nn(x))
        return self.ln2(x)


class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        seq_length,
        input_size=512,
        hidden_size=64,
        heads=8,
        enable_pos_encoding=False,
        mask=True,
    ):
        super(DecoderBlock, self).__init__()
        self.sa = MultiHeadedSelfAttention(input_size, hidden_size, heads, mask=mask)
        self.ln1 = torch.nn.LayerNorm(input_size)
        self.eda = MultiHeadedEncoderDecoderAttention(input_size, hidden_size, heads)
        self.ln2 = torch.nn.LayerNorm(input_size)
        self.nn = TransformerNN(input_size)
        self.ln3 = torch.nn.LayerNorm(input_size)
        self.dropout = torch.nn.Dropout(p=0.1)

        if enable_pos_encoding:
            self.pos_encoder = FourierEncoder2d(0.4, input_size, seq_length)
        else:
            self.pos_encoder = lambda x: x

    def forward(self, x, hidden):
        x = self.pos_encoder(x)
        x += self.dropout(self.sa(x))
        x = self.ln1(x)
        x += self.dropout(self.eda(x, hidden))
        x = self.ln2(x)
        x += self.dropout(self.nn(x))
        return self.ln3(x)


class Transformer(torch.nn.Module):
    def __init__(
        self,
        enc_count,
        dec_count,
        enc_seq_length,
        dec_seq_length,
        embedding_size=512,
        hidden_size=64,
        heads=8,
    ):
        super(Transformer, self).__init__()
        self.encoder = torch.nn.ModuleList(
            EncoderBlock(
                enc_seq_length,
                embedding_size,
                hidden_size,
                heads,
                enable_pos_encoding=(index == 0),
            )
            for index in range(enc_count)
        )
        self.decoder = torch.nn.ModuleList(
            DecoderBlock(
                dec_seq_length,
                embedding_size,
                hidden_size,
                heads,
                enable_pos_encoding=(index == 0),
            )
            for index in range(dec_count)
        )

    def forward(self, x, y):
        z = x.clone()
        for enc in self.encoder:
            z = enc(z)
        o = y
        for dec in self.decoder:
            o = dec(o, z)
        return o
