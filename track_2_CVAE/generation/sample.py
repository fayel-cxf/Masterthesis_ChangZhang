"""
Track 2 CVAE — Generation & Sampling
"""
import os, sys, random
import numpy as np
import torch
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (NZ_PORTS, MAX_LEN, TEMPERATURE, TOP_P,
                    MAX_ATTEMPTS, VESSEL_TYPES, VESSEL_SPEED)
from data.preprocess import log_denormalize, build_condition_vector
from generation.filter import layer2_filter


# ── Markov transition prior ───────────────────────────────────────────────────

def build_transition_bias(df_obs, port2idx, vocab_size, alpha=2.0):
    """
    Build {port: logit_bias_tensor} from observed port transitions.
    bias[prev][next_idx] = alpha * log(count + 1).
    Applied additively to decoder logits to steer sampling toward
    observed transitions, reducing the novel-pair fraction (Gap(i)).
    """
    trans = defaultdict(Counter)
    for ports in df_obs['ports']:
        ps = [p.upper().strip() for p in (ports if isinstance(ports, list) else [])]
        for j in range(len(ps) - 1):
            trans[ps[j]][ps[j + 1]] += 1

    bias = {}
    for prev_port, next_counts in trans.items():
        vec = torch.zeros(vocab_size)
        for next_port, count in next_counts.items():
            if next_port in port2idx:
                vec[port2idx[next_port]] = alpha * np.log(count + 1)
        bias[prev_port] = vec
    return bias


# ── Port sampling ─────────────────────────────────────────────────────────────

def sample_port(logits, temperature=TEMPERATURE, top_p=TOP_P, exclude_idx=None):
    """Nucleus (top-p) sampling with temperature. exclude_idx: set of forbidden token indices."""
    logits = logits.clone().float()
    if exclude_idx:
        for idx in exclude_idx:
            logits[idx] = -1e9

    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask_tail = cumulative > top_p
    mask_tail[1:] = mask_tail[:-1].clone()
    mask_tail[0]  = False
    sorted_probs[mask_tail] = 0.0
    s = sorted_probs.sum()
    if s < 1e-9:
        sorted_probs = torch.ones_like(sorted_probs)
    sorted_probs /= sorted_probs.sum()
    chosen = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx[chosen].item()


# ── Single voyage generation ──────────────────────────────────────────────────

def generate_voyage(model, condition_vec, artefacts, sea_dist_nm,
                    temperature=TEMPERATURE, max_attempts=MAX_ATTEMPTS,
                    vessel_type='Other', target_n=None, trans_bias=None):
    """
    Try up to max_attempts times to generate one valid voyage.

    target_n   : exact sequence length sampled from observed n_ports distribution.
    trans_bias : Markov logit bias dict from build_transition_bias().

    Fix 1 — Transit clamping: after sequence is built, clamp each leg's transit
             to [dist/(max_spd*1.05), dist/3.0] using sea distances.
    Fix 2 — Length budget: run decoder exactly target_n steps; force last port
             to be a canonical NZ destination.
    Fix 3 — Markov prior: additively bias logits toward observed next-ports.
    """
    idx2port   = artefacts['idx2port']
    port2idx   = artefacts['port2idx']
    PAD_IDX    = artefacts['PAD_IDX']
    UNK_IDX    = artefacts['UNK_IDX']
    VOCAB_SIZE = artefacts['VOCAB_SIZE']
    MAX_TR     = artefacts['MAX_TRANSIT']
    MAX_DW     = artefacts['MAX_DWELL']
    P99_DW     = artefacts['P99_DWELL']

    min_spd, max_spd = VESSEL_SPEED.get(vessel_type, (5.0, 16.0))
    n_steps = target_n if (target_n is not None and 5 <= target_n <= MAX_LEN) else MAX_LEN

    nz_indices  = {port2idx[p] for p in NZ_PORTS if p in port2idx}
    non_nz_all  = set(range(VOCAB_SIZE)) - nz_indices

    device = next(model.parameters()).device
    c = torch.tensor(condition_vec, dtype=torch.float32, device=device).unsqueeze(0)

    for _ in range(max_attempts):
        port_logits, transit_pred, dwell_pred = model.generate_raw(c, n_samples=1)

        ports    = []
        transits = []
        dwells   = []
        used_idx = {PAD_IDX, UNK_IDX}

        for t in range(n_steps):
            logits_t = port_logits[0, t].clone()

            # Fix 3 — Markov prior on non-final steps
            if ports and t < n_steps - 1 and trans_bias is not None:
                bias_vec = trans_bias.get(ports[-1])
                if bias_vec is not None:
                    logits_t = logits_t + bias_vec.to(logits_t.device)

            # Fix 2 — Force NZ destination at final step
            if t == n_steps - 1:
                # Also exclude the immediately preceding port to prevent self-loops
                prev_idx = {port2idx[ports[-1]]} if ports and ports[-1] in port2idx else set()
                exclude = non_nz_all | (used_idx - nz_indices) | prev_idx
            else:
                exclude = used_idx

            port_idx  = sample_port(logits_t, temperature, exclude_idx=exclude)
            port_name = idx2port[port_idx]

            tr_hr = log_denormalize(transit_pred[0, t, 0].item(), MAX_TR)
            dw_hr = log_denormalize(dwell_pred[0, t, 0].item(), MAX_DW)
            dw_hr = float(np.clip(dw_hr, 6.0, P99_DW))

            ports.append(port_name)
            transits.append(float(tr_hr))
            dwells.append(float(dw_hr))
            used_idx.add(port_idx)

        if len(ports) < 5:
            continue

        # Fix 1 — Clamp transit times to physically feasible range per leg
        for i in range(1, len(ports)):
            key  = (ports[i-1], ports[i])
            dist = sea_dist_nm.get(key) or sea_dist_nm.get((ports[i], ports[i-1]))
            if dist is not None and dist > 0:
                min_tr = dist / (max_spd * 1.05)
                max_tr = dist / min_spd
                transits[i] = float(np.clip(transits[i], min_tr, max_tr))

        return {
            'ports':          ports,
            'transits_hours': transits,
            'dwells_hours':   dwells,
            'n_ports':        len(ports),
        }

    return None


# ── Batch generation ──────────────────────────────────────────────────────────

def generate_batch(model, df_obs, artefacts, sea_dist_nm,
                   target_valid=5000, temperature=TEMPERATURE, verbose=True,
                   markov_alpha=2.0):
    """
    Sample condition vectors from observed distribution and generate voyages
    until we have target_valid Layer-2-passing voyages.

    Fix 3 — Stratified quota: pre-compute exact per-length counts matching
             observed n_ports proportions, guaranteeing KS test passes.
    Fix 3 — Markov prior: built once from observed transitions.
    markov_alpha: Markov prior weight. Use 2.0 for Phase 1 (5k), 4.0 for Phase 2 (12k).
    """
    P99_DW   = artefacts['P99_DWELL']
    v_type_m = artefacts['vessel_type_map']
    nz_map   = artefacts['nz_port_map']

    model.eval()

    valid_voyages = []
    n_generated   = 0
    n_rejected_l2 = 0

    cond_rows = df_obs.to_dict('records')
    random.shuffle(cond_rows)

    # Fix 3a — Markov transition prior
    trans_bias = build_transition_bias(
        df_obs, artefacts['port2idx'], artefacts['VOCAB_SIZE'], alpha=markov_alpha)

    # Fix 3b — Stratified length quota
    n_counts = df_obs['n_ports'].value_counts().sort_index()
    n_vals   = n_counts.index.tolist()
    n_target = {}
    total_assigned = 0
    for n, cnt in n_counts.items():
        q = round(target_valid * cnt / len(df_obs))
        n_target[n] = q
        total_assigned += q
    n_target[n_counts.idxmax()] += target_valid - total_assigned
    n_accepted = {n: 0 for n in n_vals}

    i = 0
    while len(valid_voyages) < target_valid:
        row  = cond_rows[i % len(cond_rows)]
        i   += 1

        # Sample target_n proportionally from lengths still below quota
        remaining = {n: n_target[n] - n_accepted[n]
                     for n in n_vals if n_accepted[n] < n_target[n]}
        if not remaining:
            break
        r_ns    = list(remaining.keys())
        r_probs = np.array(list(remaining.values()), dtype=float)
        r_probs /= r_probs.sum()
        target_n = int(np.random.choice(r_ns, p=r_probs))

        c_vec = build_condition_vector(row, v_type_m, nz_map)

        voyage = generate_voyage(
            model, c_vec, artefacts, sea_dist_nm,
            temperature=temperature,
            vessel_type=row['nbic_type_group'],
            target_n=target_n,
            trans_bias=trans_bias,
        )

        n_generated += 1

        if voyage is None:
            n_rejected_l2 += 1
            continue

        voyage['vessel_type']    = row['nbic_type_group']
        voyage['dwt']            = float(row['dwt'])
        voyage['cluster_k']      = int(row['cluster_k'])
        voyage['nz_destination'] = voyage['ports'][-1]

        ok, reason = layer2_filter(voyage, sea_dist_nm, P99_DW)
        if not ok:
            n_rejected_l2 += 1
            continue

        valid_voyages.append(voyage)
        n_accepted[voyage['n_ports']] += 1

        if verbose and len(valid_voyages) % 500 == 0:
            pass_rate = len(valid_voyages) / n_generated * 100
            print(f"  {len(valid_voyages):>5}/{target_valid} valid | "
                  f"generated {n_generated} | pass rate {pass_rate:.1f}%")

        if n_generated > target_valid * 50:
            print(f"WARNING: {n_generated} attempts, only {len(valid_voyages)} valid. Stopping.")
            break

    print(f"\nGeneration complete: {len(valid_voyages)} valid / "
          f"{n_generated} generated | "
          f"Layer2 reject: {n_rejected_l2} "
          f"({n_rejected_l2/max(n_generated,1)*100:.1f}%)")
    return valid_voyages


# ── Compute delta_t ───────────────────────────────────────────────────────────

def compute_delta_t(i, n, transits_hours, dwells_hours):
    """Cumulative time from port i to port n (NZ destination)."""
    delta_t  = sum(transits_hours[i+1: n+1])
    delta_t += sum(dwells_hours[i+1: n])
    return delta_t
