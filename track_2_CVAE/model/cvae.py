"""
Track 2 CVAE — Full Model (Encoder + Decoder + VoyageCVAE)
"""
import numpy as np
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBED_DIM, HIDDEN_DIM, LATENT_DIM, CONDITION_DIM, GRU_LAYERS, GRU_DROPOUT, EMBED_DROPOUT, MAX_LEN


# ── Geo-coordinate embedding initialiser ─────────────────────────────────────

def init_embedding_from_coords(port2idx, port_coords, embed_dim=32):
    vocab_size = len(port2idx)
    weight = np.random.randn(vocab_size, embed_dim) * 0.01
    n_init = 0
    for port, idx in port2idx.items():
        if port in port_coords:
            lat, lon = port_coords[port]
            weight[idx, 0] = np.sin(np.radians(lat))
            weight[idx, 1] = np.cos(np.radians(lat))
            weight[idx, 2] = np.sin(np.radians(lon))
            weight[idx, 3] = np.cos(np.radians(lon))
            n_init += 1
    weight[0] = 0.0   # PAD_IDX=0 → zero
    print(f"Geo-init: {n_init}/{vocab_size} ports initialised from coordinates.")
    return torch.tensor(weight, dtype=torch.float32)


# ── Reparameterisation ────────────────────────────────────────────────────────

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, embedding, embed_dim, condition_dim,
                 hidden_dim, latent_dim, gru_layers=2, gru_dropout=0.3,
                 embed_dropout=0.1):
        super().__init__()
        self.port_embedding  = embedding
        self.embed_dropout   = nn.Dropout(p=embed_dropout)
        self.gru = nn.GRU(
            input_size=embed_dim + 2,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )
        self.fc_mu      = nn.Linear(hidden_dim + condition_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim + condition_dim, latent_dim)

    def forward(self, port_indices, time_feats, c):
        port_emb = self.embed_dropout(self.port_embedding(port_indices))
        x_seq    = torch.cat([port_emb, time_feats], dim=-1)
        _, h_n   = self.gru(x_seq)
        h_last   = h_n[-1]
        combined = torch.cat([h_last, c], dim=-1)
        return self.fc_mu(combined), self.fc_log_var(combined)


# ── Decoder ───────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, embedding, embed_dim, latent_dim,
                 condition_dim, hidden_dim, vocab_size, max_len,
                 gru_layers=2, gru_dropout=0.3):
        super().__init__()
        self.max_len        = max_len
        self.vocab_size     = vocab_size
        self.port_embedding = embedding
        self.fc_init        = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.gru_layers     = gru_layers
        self.gru = nn.GRU(
            input_size=embed_dim + 2,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )
        self.port_head    = nn.Linear(hidden_dim, vocab_size)
        self.transit_head = nn.Linear(hidden_dim, 1)
        self.dwell_head   = nn.Linear(hidden_dim, 1)

    def forward(self, z, c, true_port_indices=None, true_time_feats=None,
                teacher_forcing_ratio=1.0):
        batch_size = z.shape[0]
        embed_dim  = self.port_embedding.embedding_dim
        device     = z.device

        h = torch.tanh(self.fc_init(torch.cat([z, c], dim=-1)))
        h = h.unsqueeze(0).repeat(self.gru_layers, 1, 1)

        step_input = torch.zeros(batch_size, embed_dim + 2, device=device)

        port_logits_all, transit_all, dwell_all = [], [], []

        for t in range(self.max_len):
            out, h = self.gru(step_input.unsqueeze(1), h)
            out    = out.squeeze(1)

            port_logits = self.port_head(out)
            transit     = torch.sigmoid(self.transit_head(out))
            dwell       = torch.sigmoid(self.dwell_head(out))

            port_logits_all.append(port_logits)
            transit_all.append(transit)
            dwell_all.append(dwell)

            if true_port_indices is not None:
                use_tf = (torch.rand(1).item() < teacher_forcing_ratio)
                if use_tf:
                    port_emb_next = self.port_embedding(true_port_indices[:, t])
                    time_next     = true_time_feats[:, t, :]
                else:
                    port_probs    = torch.softmax(port_logits.detach(), dim=-1)
                    port_emb_next = port_probs @ self.port_embedding.weight
                    time_next     = torch.cat([transit.detach(),
                                               dwell.detach()], dim=-1)
            else:
                # generation mode
                port_probs    = torch.softmax(port_logits, dim=-1)
                port_emb_next = port_probs @ self.port_embedding.weight
                time_next     = torch.cat([transit, dwell], dim=-1)

            step_input = torch.cat([port_emb_next, time_next], dim=-1)

        return (
            torch.stack(port_logits_all, dim=1),  # (B, MAX_LEN, vocab)
            torch.stack(transit_all,     dim=1),  # (B, MAX_LEN, 1)
            torch.stack(dwell_all,       dim=1),  # (B, MAX_LEN, 1)
        )


# ── Full CVAE ─────────────────────────────────────────────────────────────────

class VoyageCVAE(nn.Module):
    def __init__(self, vocab_size, port2idx=None, port_coords=None,
                 embed_dim=EMBED_DIM, condition_dim=CONDITION_DIM,
                 hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
                 max_len=MAX_LEN, gru_layers=GRU_LAYERS,
                 gru_dropout=GRU_DROPOUT, embed_dropout=EMBED_DROPOUT):
        super().__init__()
        self.latent_dim = latent_dim

        self.shared_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,   # PAD_IDX always 0
        )
        if port2idx is not None and port_coords is not None:
            init_w = init_embedding_from_coords(port2idx, port_coords, embed_dim)
            self.shared_embedding.weight = nn.Parameter(init_w)

        self.encoder = Encoder(
            embedding=self.shared_embedding,
            embed_dim=embed_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            gru_layers=gru_layers,
            gru_dropout=gru_dropout,
            embed_dropout=embed_dropout,
        )
        self.decoder = Decoder(
            embedding=self.shared_embedding,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            max_len=max_len,
            gru_layers=gru_layers,
            gru_dropout=gru_dropout,
        )

    def forward(self, port_indices, time_feats, c,
                teacher_forcing_ratio=1.0):
        mu, log_var = self.encoder(port_indices, time_feats, c)
        z           = reparameterize(mu, log_var)
        port_logits, transit_pred, dwell_pred = self.decoder(
            z, c,
            true_port_indices=port_indices,
            true_time_feats=time_feats,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return port_logits, transit_pred, dwell_pred, mu, log_var

    @torch.no_grad()
    def generate_raw(self, c, n_samples=1):
        """Sample z ~ N(0,I) and decode. Returns raw logits/preds."""
        device = next(self.parameters()).device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        if c.shape[0] == 1 and n_samples > 1:
            c = c.expand(n_samples, -1)
        port_logits, transit_pred, dwell_pred = self.decoder(z, c)
        return port_logits, transit_pred, dwell_pred
