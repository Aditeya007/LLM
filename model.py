import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTConfig:
    vocab_size = 50257
    block_size = 512
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.1
    bias = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias
        )

        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_dim = C // self.n_head

        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd,
            bias=config.bias
        )

        self.gelu = nn.GELU()

        self.c_proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd,
            bias=config.bias
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.vocab_size,
                    config.n_embd
                ),
                wpe=nn.Embedding(
                    config.block_size,
                    config.n_embd
                ),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        n_params = sum(
            p.numel()
            for p in self.parameters()
        )

        print(
            f"Model initialized: "
            f"{n_params / 1e6:.1f}M parameters"
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, idx, targets=None):
        B, T = idx.size()

        assert T <= self.config.block_size

        pos = torch.arange(
            0,
            T,
            dtype=torch.long,
            device=idx.device
        )

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(
            tok_emb + pos_emb
        )

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None
    ):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )

                logits[
                    logits < v[:, [-1]]
                ] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx