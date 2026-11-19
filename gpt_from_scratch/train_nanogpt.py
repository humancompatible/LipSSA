import time

import torch
import torch.nn as nn
from torch.nn import functional as F

block_size, n_embd, n_head, n_layer, dropout = 256, 384, 6, 6, 0.2
batch_size, max_iters, eval_interval, eval_iters, learning_rate = 64, 5000, 500, 200, 3e-4
OUT = f'model_weights_{n_layer}x{n_embd}.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
splits = {'train': data[:n], 'val': data[n:]}


def get_batch(split):
    d = splits[split]
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def embed(self, idx):
        return self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(idx.shape[1], device=idx.device))
    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def loss_of(model, x, y):
    logits = model(model.embed(x))
    return F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {s: torch.stack([loss_of(model, *get_batch(s)) for _ in range(eval_iters)]).mean().item() for s in splits}
    model.train()
    return out


model = BigramLanguageModel().to(device)
print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters on {device}', flush=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start = time.time()
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}, {time.time() - start:.0f} s", flush=True)
    if it == max_iters:
        break
    loss = loss_of(model, *get_batch('train'))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

config = dict(block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, vocab_size=vocab_size)
torch.save({'config': config, 'state_dict': model.cpu().state_dict(), 'losses': losses}, OUT)
print(f'saved {OUT}', flush=True)
