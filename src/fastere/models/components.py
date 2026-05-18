import torch
import torch.nn as nn


class MultiNonLinearClassifier(nn.Module):
    def __init__(
        self,
        hidden_size,
        tag_size,
        layers_num=1,
        hidden_dim=None,
        dropout_rate=0.1,
        with_init=False,
        use_norm=False,
        return_hidden_size=False,
    ):
        super().__init__()
        self.tag_size = tag_size
        self.return_hidden_size = return_hidden_size

        if hidden_dim is None:
            hidden_dim = hidden_size // 2

        input_dims = [hidden_size] + [hidden_dim] * (layers_num - 1)
        output_dims = [hidden_dim] * layers_num

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dims[i], output_dims[i]),
                    nn.LayerNorm(output_dims[i]) if use_norm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                )
                for i in range(layers_num)
            ]
        )
        self.classifier = nn.Linear(output_dims[-1], tag_size)

        if with_init:
            self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if "weight" in name:
                    nn.init.xavier_normal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
        for name, param in self.classifier.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.classifier(x)
        if self.return_hidden_size:
            return out, x
        return out


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, ent_range=None):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, bias=True, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ent_range = ent_range

    def forward(self, x):
        if self.ent_range:
            attn_mask = torch.ones(x.shape[1], x.shape[1], device=x.device)
            attn_mask = torch.tril(attn_mask, diagonal=self.ent_range) * torch.triu(
                attn_mask, diagonal=-self.ent_range
            )
        else:
            attn_mask = None
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask)
        return self.layer_norm(x + attn_output)


class FeedForward(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dims, activations, dropout):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * num_layers
        if isinstance(activations, (str,)) or callable(activations):
            activations = [activations] * num_layers
        if isinstance(dropout, (float, int)):
            dropout = [dropout] * num_layers

        input_dims_list = [input_dim] + hidden_dims[:-1]
        self.linear_layers = nn.ModuleList(
            nn.Linear(i, o) for i, o in zip(input_dims_list, hidden_dims)
        )
        self.activations = []
        for a in activations:
            if a == "relu":
                self.activations.append(nn.ReLU())
            elif a == "gelu":
                self.activations.append(nn.GELU())
            elif callable(a):
                self.activations.append(a)
            else:
                raise ValueError(f"Invalid activation: {a}")
        self.dropout = nn.ModuleList(nn.Dropout(p=d) for d in dropout)

    def forward(self, x):
        for layer, activation, drop in zip(
            self.linear_layers, self.activations, self.dropout
        ):
            x = drop(activation(layer(x)))
        return x


class PrefixEncoder(torch.nn.Module):
    """Encodes soft prefix tokens for prefix-tuning."""

    def __init__(
        self,
        num_hidden_layers,
        prompt_len,
        prefix_hidden_size,
        hidden_size,
        prefix_projection=False,
    ):
        super().__init__()
        self.prefix_projection = prefix_projection
        if self.prefix_projection:
            self.embedding = torch.nn.Embedding(prompt_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(
                    prefix_hidden_size, num_hidden_layers * 2 * hidden_size
                ),
            )
        else:
            self.embedding = torch.nn.Embedding(
                prompt_len, num_hidden_layers * 2 * hidden_size
            )

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            return self.trans(self.embedding(prefix))
        return self.embedding(prefix)
