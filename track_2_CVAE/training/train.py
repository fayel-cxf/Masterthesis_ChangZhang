"""
Track 2 CVAE — Training Loop
"""
import os, sys, pickle, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (BATCH_SIZE, TOTAL_EPOCHS, LR, WEIGHT_DECAY,
                    KL_WARMUP, KL_BETA_MAX, GRAD_CLIP, LAMBDA1, LAMBDA2,
                    VAL_SPLIT, OUT_DIR)
from data.preprocess import preprocess
from data.dataset    import VoyageDataset
from model.cvae      import VoyageCVAE


# ── Loss ──────────────────────────────────────────────────────────────────────

def cvae_loss(port_logits, transit_pred, dwell_pred,
              port_target, transit_target, dwell_target,
              mu, log_var, mask, lambda1, lambda2, beta):
    B, L, V = port_logits.shape

    L_port = nn.CrossEntropyLoss(reduction='none')(
        port_logits.reshape(-1, V),
        port_target.reshape(-1),
    ).reshape(B, L)
    L_port = (L_port * mask).sum() / mask.sum()

    L_transit = nn.MSELoss(reduction='none')(
        transit_pred.squeeze(-1), transit_target)
    L_transit = (L_transit * mask).sum() / mask.sum()

    L_dwell = nn.MSELoss(reduction='none')(
        dwell_pred.squeeze(-1), dwell_target)
    L_dwell = (L_dwell * mask).sum() / mask.sum()

    L_KL = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    total = L_port + lambda1 * L_transit + lambda2 * L_dwell + beta * L_KL
    return total, L_port, L_transit, L_dwell, L_KL


# ── Schedulers ────────────────────────────────────────────────────────────────

def get_beta(epoch, total_epochs, warmup_ratio=KL_WARMUP, beta_max=KL_BETA_MAX):
    warmup = int(total_epochs * warmup_ratio)
    if epoch < warmup:
        return beta_max * (epoch / max(warmup, 1))
    return beta_max


def get_tf_ratio(epoch, total_epochs):
    if epoch < total_epochs * 0.5:
        return 1.0
    progress = (epoch - total_epochs * 0.5) / (total_epochs * 0.5)
    return max(0.5, 1.0 - 0.5 * progress)


# ── Port accuracy helper ──────────────────────────────────────────────────────

def port_accuracy(port_logits, port_target, mask):
    preds   = port_logits.argmax(dim=-1)  # (B, L)
    correct = ((preds == port_target) * mask.bool()).sum().float()
    total   = mask.sum()
    return (correct / total).item() if total > 0 else 0.0


# ── Main training function ────────────────────────────────────────────────────

def train(lambda1=LAMBDA1, lambda2=LAMBDA2,
          total_epochs=TOTAL_EPOCHS, seed=42):

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    df, artefacts = preprocess(save_pkl=True)

    dataset   = VoyageDataset(df, artefacts)
    n_val     = max(1, int(len(dataset) * VAL_SPLIT))
    n_train   = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"Train: {n_train} | Val: {n_val}")

    # ── Model ──────────────────────────────────────────────────────────────────
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = VoyageCVAE(
        vocab_size  = artefacts['VOCAB_SIZE'],
        port2idx    = artefacts['port2idx'],
        port_coords = artefacts['port_coords'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-5)

    history = []
    best_val_loss = float('inf')
    model_path = os.path.join(OUT_DIR, 'best_model.pt')

    t0 = time.time()
    for epoch in range(total_epochs):
        beta     = get_beta(epoch, total_epochs)
        tf_ratio = get_tf_ratio(epoch, total_epochs)

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        ep_losses = []
        for batch in train_loader:
            (port_idx, time_f, c_vec,
             port_tgt, tr_tgt, dw_tgt, mask) = [b.to(device) for b in batch]

            optimizer.zero_grad()
            pl, tp, dp, mu, lv = model(port_idx, time_f, c_vec,
                                        teacher_forcing_ratio=tf_ratio)
            loss, lp, lt, ld, lkl = cvae_loss(
                pl, tp, dp, port_tgt, tr_tgt, dw_tgt,
                mu, lv, mask, lambda1, lambda2, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            ep_losses.append({'total': loss.item(), 'port': lp.item(),
                               'transit': lt.item(), 'dwell': ld.item(),
                               'kl': lkl.item()})

        train_avg = {k: np.mean([x[k] for x in ep_losses])
                     for k in ep_losses[0]}

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for batch in val_loader:
                (port_idx, time_f, c_vec,
                 port_tgt, tr_tgt, dw_tgt, mask) = [b.to(device) for b in batch]
                pl, tp, dp, mu, lv = model(port_idx, time_f, c_vec,
                                            teacher_forcing_ratio=1.0)
                vloss, *_ = cvae_loss(
                    pl, tp, dp, port_tgt, tr_tgt, dw_tgt,
                    mu, lv, mask, lambda1, lambda2, beta)
                val_losses.append(vloss.item())
                val_accs.append(port_accuracy(pl, port_tgt, mask))

        val_loss = np.mean(val_losses)
        val_acc  = np.mean(val_accs)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state': model.state_dict(),
                        'artefacts': artefacts,
                        'epoch': epoch,
                        'val_loss': val_loss}, model_path)

        rec = dict(epoch=epoch, beta=beta, tf=tf_ratio,
                   val_loss=val_loss, val_acc=val_acc, **train_avg)
        history.append(rec)

        if epoch % 10 == 0 or epoch == total_epochs - 1:
            elapsed = time.time() - t0
            print(f"Ep {epoch:03d} | β={beta:.2f} tf={tf_ratio:.2f} | "
                  f"train={train_avg['total']:.3f} port={train_avg['port']:.3f} "
                  f"KL={train_avg['kl']:.2f} | "
                  f"val={val_loss:.3f} acc={val_acc:.2%} | "
                  f"{elapsed/60:.1f}min")

        # early stop if port loss stops improving (very rough guard)
        if epoch >= 20 and train_avg['port'] < 0.05:
            print("Port loss < 0.05 — early convergence, stopping.")
            break

    # Save history
    hist_path = os.path.join(OUT_DIR, 'training_history.pkl')
    with open(hist_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"\nTraining done. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {model_path}")

    return model, artefacts, history


if __name__ == '__main__':
    train()
