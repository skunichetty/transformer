import torch


class Attention(torch.nn.Module):
    """Generic attention module with support for optional masking"""

    def __init__(self):
        """Initialize the attention module"""
        super(Attention, self).__init__()

    def forward(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        mask: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Perform attention operation of query, key, and value vectors

        Args:
            q (torch.FloatTensor): Query vectors - compared against key vectors to generate weights for weighted average
            k (torch.FloatTensor): Key vectors - compared with query vectors to generate weights for weighted average
            v (torch.FloatTensor): Value vectors - vectors to take a weighted average of using weights generated from query and key vectors
            mask (torch.FloatTensor, optional): Mask to be applied to weights before weights average. Defaults to None, in which case no mask is applied

        Returns:
            torch.FloatTensor: Output of attention operation
        """
        a = q @ k.T
        if mask is not None:
            a += mask
        return self.softmax(a / torch.sqrt(self.output_size)) @ v

    def __repr__(self):
        return "Attention"


class SelfAttention(torch.nn.Module):
    """Generic self-attention module for use in a transformer"""

    def __init__(
        self, input_size: int = 512, output_size: int = 64, mask: bool = False
    ):
        """
        Initialize self-attention module.

        Args:
            input_size (int, optional): Dimension of input vectors. Defaults to 512.
            output_size (int, optional): Dimension of output vectors. Defaults to 64.
            mask (bool, optional): Specifies whether to mask attention weights to prevent look-ahead. Defaults to False.
        """
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
        """Initialize module weights using the specified distribution."""
        weights = (self.wquery, self.wkey, self.wvalue)
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="relu")

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Perform self attention operation

        Args:
            x (torch.FloatTensor): Input sequence of vectors (such as word embeddings for a sequence)

        Returns:
            torch.FloatTensor: Output sequence after self-attention operation is applied
        """
        q, k, v = x @ self.wquery, x @ self.wkey, x @ self.wvalue
        if self.mask:
            return self.attn(q, k, v, torch.triu(q @ k.T) * float("-inf"))
        else:
            return self.attn(q, k, v)

    def __repr__(self):
        return "SelfAttention"


class MultiHeadedSelfAttention(torch.nn.Module):
    """Multi-headed self attention module."""

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 64,
        heads: int = 8,
        mask: bool = False,
    ):
        """
        Initialize multi-headed self-attention layer

        Args:
            input_size (int, optional): Dimension of input vectors. Defaults to 512.
            output_size (int, optional): Dimension of output vectors. Defaults to 64.
            heads (int, optional): Number of self-attention heads to use in network. Defaults to 8.
            mask (bool, optional): Specifies whether to mask attention weights to prevent look-ahead. Defaults to False.
        """
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.heads = torch.nn.ModuleList(
            [SelfAttention(input_size, hidden_size, mask) for _ in range(heads)]
        )
        self.w0 = torch.nn.Parameter(torch.empty(heads * hidden_size, input_size))
        torch.nn.init.kaiming_normal_(self.w0, nonlinearity="relu")

    def forward(self, x):
        """
        Perform multi-headed self-attention on input sequence of vectors. The output from each head is recombined
        using a simple dense layer of neurons.

        Args:
            x (torch.FloatTensor): Input sequence of vectors (such as word embeddings for a sequence)

        Returns:
            torch.FloatTensor: Output sequence after multi-headed self-attention operation is applied
        """
        embeddings = tuple(map(lambda a: a(x), self.heads))
        concatenated = torch.cat(embeddings, dim=1).to(x.device)
        return concatenated @ self.w0

    def __repr__(self):
        return f"MultiHeadedSelfAttention(embedding_dim={self.input_size}, hidden_dim={self.hidden_size}, heads={len(self.heads)})"


class EncoderDecoderAttention(torch.nn.Module):
    """Encoder-Decoder attention module, as specified in "Attention is all you need"."""

    def __init__(self, input_size: int = 512, output_size: int = 64):
        """
        Initialize encoder-decoder attention module.

        Args:
            input_size (int, optional): Dimension of input vectors. Defaults to 512.
            output_size (int, optional): Dimension of output vectors. Defaults to 64.
        """
        super(EncoderDecoderAttention, self).__init__()

        self.wquery = torch.nn.Parameter(torch.empty(input_size, output_size))
        self.wkey = torch.nn.Parameter(torch.empty(input_size, output_size))
        self.wvalue = torch.nn.Parameter(torch.empty(input_size, output_size))
        self.init_weights()

        self.attn = Attention()

    def init_weights(self):
        """Initialize module weights using the specified distribution."""
        weights = (self.wquery, self.wkey, self.wvalue)
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="relu")

    def forward(
        self, x: torch.FloatTensor, embeddings: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Perform encoder-decoder attention operation using query from previous layer, keys and values from encoder output.

        Args:
            x (torch.FloatTensor): Input sequence of vectors (such as word embeddings for a sequence)
            embeddings (torch.FloatTensor): Embedded sequence of vectors from output of encoder

        Returns:
            torch.FloatTensor: Output sequence after encoder-decoder attention operation is applied
        """
        q, k, v = x @ self.wquery, embeddings @ self.wkey, embeddings @ self.wvalue
        return self.attn(q, k, v)

    def __repr__(self):
        return "EncoderDecoderAttention"


class MultiHeadedEncoderDecoderAttention(torch.nn.Module):
    """Multiheaded encoder-decoder attention module."""

    def __init__(self, input_size: int = 512, hidden_size: int = 64, heads: int = 8):
        """
        Initialize multi-headed encoder-decoder attention layer

        Args:
            input_size (int, optional): Dimension of input vectors. Defaults to 512.
            output_size (int, optional): Dimension of output vectors. Defaults to 64.
            heads (int, optional): Number of encoder-decoder attention heads to use in network. Defaults to 8.
        """
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
        """
        Perform encoder-decoder attention operation using query from previous layer, keys and values from encoder output.

        Args:
            x (torch.FloatTensor): Input sequence of vectors (such as word embeddings for a sequence)
            embeddings (torch.FloatTensor): Embedded sequence of vectors from output of encoder

        Returns:
            torch.FloatTensor: Output sequence after encoder-decoder attention operation is applied
        """
        embeddings = tuple(map(lambda a: a(x, hidden), self.heads))
        concatenated = torch.cat(embeddings, dim=1).to(x.device)
        return concatenated @ self.w0

    def __repr__(self):
        return f"MultiHeadedEncoderDecoderAttention(embedding_dim={self.input_size}, hidden_dim={self.hidden_size}, heads={len(self.heads)})"


class TransformerNN(torch.nn.Module):
    """Basic feedforward neural net used in encoder and decoder block in "Attention is All You Need"."""

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
    """
    Module that generates positional embeddings for the input sequence.

    Necessary for adding input order back into attention operation.
    """

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


class Encoder(torch.nn.Module):
    def __init__(
        self,
        seq_length,
        embedding_size=512,
        hidden_size=64,
        heads=8,
        enable_pos_encoding=False,
    ):
        super(Encoder, self).__init__()
        self.sa = SelfAttention(embedding_size, hidden_size, heads)
        self.ln1 = torch.nn.LayerNorm(embedding_size)
        self.nn = TransformerNN(embedding_size)
        self.ln2 = torch.nn.LayerNorm(embedding_size)
        self.encode_pos = enable_pos_encoding
        self.dropout = torch.nn.Dropout(p=0.1)
        self.pos_encoder = FourierEncoder2d(0.4, embedding_size, seq_length)

    def forward(self, x):
        if self.encode_pos:
            x = self.pos_encoder(x)
        x += self.dropout(self.sa(x))
        x = self.ln1(x)
        x += self.dropout(self.nn(x))
        return self.ln2(x)


class Decoder(torch.nn.Module):
    def __init__(
        self,
        seq_length,
        embedding_size=512,
        hidden_size=64,
        heads=8,
        enable_pos_encoding=False,
        mask=True,
    ):
        super(Decoder, self).__init__()
        self.sa = SelfAttention(embedding_size, hidden_size, heads, mask=True)
        self.ln1 = torch.nn.LayerNorm(embedding_size)
        self.eda = MultiHeadedEncoderDecoderAttention(
            embedding_size, hidden_size, heads
        )
        self.ln2 = torch.nn.LayerNorm(embedding_size)
        self.nn = TransformerNN(embedding_size)
        self.ln3 = torch.nn.LayerNorm(embedding_size)
        self.encode_pos = enable_pos_encoding
        self.dropout = torch.nn.Dropout(p=0.1)
        self.pos_encoder = FourierEncoder2d(0.4, embedding_size, seq_length)

    def forward(self, x, hidden):
        if self.encode_pos:
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
        *args,
        **kwargs,
    ):
        super(Transformer, self).__init__(*args, **kwargs)
        self.enc_stack = torch.nn.ModuleList(
            [
                Encoder(enc_seq_length, embedding_size, hidden_size, heads)
                for _ in range(enc_count)
            ]
        )
        self.enc_stack[0].encode_pos = True
        self.dec_stack = torch.nn.ModuleList(
            [
                Decoder(dec_seq_length, embedding_size, hidden_size, heads)
                for _ in range(dec_count)
            ]
        )
        self.dec_stack[0].encode_pos = True

    def forward(self, x, y):
        z = x.clone()
        for enc in self.enc_stack:
            z = enc(z)
        o = y
        for dec in self.dec_stack:
            o = dec(o, z)
        return o
