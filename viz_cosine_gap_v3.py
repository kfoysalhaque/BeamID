# viz_cosine_gap_cached_3panel.py

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import matplotlib as mpl

from datagen import make_splits, set_seed
from model import Encoder, Classifier

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{mathptmx}
    """
})

# ---------- Global plot style  ----------
mpl.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

@torch.no_grad()
def collect_embeddings(encoder, loader, device):
    """Return normalized embeddings Z and integer labels Y for a loader."""
    encoder.eval()
    Z_list, Y_list = [], []
    for x, y, _ in loader:
        x = x.to(device)
        z = encoder(x)
        z = F.normalize(z, dim=1)
        Z_list.append(z.cpu().numpy())
        Y_list.append(y.numpy())
    Z = np.concatenate(Z_list, axis=0) if len(Z_list) else np.zeros((0, 1), dtype=np.float32)
    Y = np.concatenate(Y_list, axis=0) if len(Y_list) else np.zeros((0,), dtype=np.int64)
    return Z, Y


def sample_pairwise_cosines(Za, Ya, Zb, Yb, n_pairs=50000, seed=0):
    """
    Randomly sample pairs across two sets (A,B) and return:
      - pos_cos: cosine similarities for same-class pairs
      - neg_cos: cosine similarities for different-class pairs
    Assumes embeddings are already L2-normalized -> cosine = dot.
    """
    rng = np.random.default_rng(seed)

    a_by_c, b_by_c = {}, {}
    for i, c in enumerate(Ya):
        a_by_c.setdefault(int(c), []).append(i)
    for i, c in enumerate(Yb):
        b_by_c.setdefault(int(c), []).append(i)

    classes = sorted(set(a_by_c.keys()) & set(b_by_c.keys()))
    if len(classes) == 0:
        return np.array([]), np.array([])

    for c in classes:
        a_by_c[c] = np.asarray(a_by_c[c], dtype=int)
        b_by_c[c] = np.asarray(b_by_c[c], dtype=int)

    # positives (roughly even over classes)
    pos = []
    per_c = max(1, n_pairs // len(classes))
    for c in classes:
        ai = rng.choice(a_by_c[c], size=per_c, replace=True)
        bi = rng.choice(b_by_c[c], size=per_c, replace=True)
        pos.append(np.sum(Za[ai] * Zb[bi], axis=1))
    pos = np.concatenate(pos, axis=0) if len(pos) else np.array([])

    # negatives (a-class != b-class)
    neg = []
    class_arr = np.asarray(classes, dtype=int)
    need = n_pairs

    # rejection-free mismatch sampling in chunks
    while len(neg) == 0 or np.concatenate(neg).shape[0] < need:
        m = max(need, n_pairs)
        ca = rng.choice(class_arr, size=m, replace=True)
        cb = rng.choice(class_arr, size=m, replace=True)
        mask = ca != cb
        ca, cb = ca[mask], cb[mask]
        if ca.size == 0:
            continue

        ai = np.array([rng.choice(a_by_c[int(c)]) for c in ca], dtype=int)
        bi = np.array([rng.choice(b_by_c[int(c)]) for c in cb], dtype=int)
        neg.append(np.sum(Za[ai] * Zb[bi], axis=1))

    neg = np.concatenate(neg, axis=0)[:need] if len(neg) else np.array([])
    return pos, neg


def fewshot_finetune(encoder, classifier, adapt_loader, device, steps=800, lr=5e-4, unfreeze_last=False):
    """
    Few-shot fine-tuning:
    - default: classifier only
    - if unfreeze_last: classifier + encoder.fc
    """
    for p in encoder.parameters():
        p.requires_grad = False
    for p in classifier.parameters():
        p.requires_grad = True

    params = list(classifier.parameters())

    if unfreeze_last:
        for p in encoder.fc.parameters():
            p.requires_grad = True
        params += list(encoder.fc.parameters())

    opt = torch.optim.Adam(params, lr=lr)

    encoder.train()
    classifier.train()

    it = iter(adapt_loader)
    for step in range(1, steps + 1):
        try:
            x, y, _ = next(it)
        except StopIteration:
            it = iter(adapt_loader)
            x, y, _ = next(it)

        x, y = x.to(device), y.to(device)
        logits = classifier(encoder(x))
        loss = F.cross_entropy(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0:
            print(f"[ft] step {step:04d}/{steps} | loss={loss.item():.4f}")


def plot_gap_pubready(
    ax,
    pos_cos,
    neg_cos,
    title,
    bins=60,
    show_fill=True,
    show_medians=True,
    show_gap_text=True,
    legend_loc="upper left",
):
    pos_cos = np.asarray(pos_cos).ravel()
    neg_cos = np.asarray(neg_cos).ravel()

    edges = np.linspace(-1.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax.set_axisbelow(True)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pos_hist, _ = np.histogram(pos_cos, bins=edges, density=True) if pos_cos.size else (np.zeros(bins), edges)
    neg_hist, _ = np.histogram(neg_cos, bins=edges, density=True) if neg_cos.size else (np.zeros(bins), edges)

    ax.step(centers, neg_hist, where="mid", linewidth=2.2, label="Diff. NIC", color="blue")
    ax.step(centers, pos_hist, where="mid", linewidth=2.2, label="Same NIC", color="red")

    if show_fill:
        ax.fill_between(centers, 0, neg_hist, step="mid", alpha=0.12)
        ax.fill_between(centers, 0, pos_hist, step="mid", alpha=0.12)

    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title(title)

    if pos_cos.size and neg_cos.size:
        med_pos = float(np.median(pos_cos))
        med_neg = float(np.median(neg_cos))
        gap = med_pos - med_neg

        if show_medians:
            ax.axvline(med_neg, linestyle="--", linewidth=2, alpha=1, color="blue")
            ax.axvline(med_pos, linestyle="--", linewidth=2, alpha=1, color="red")

        if show_gap_text:
            txt = rf"$\Delta_{{\mathrm{{median}}}}={gap:.3f}$"
            ax.text(
                0.02, 0.98, txt,
                transform=ax.transAxes,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.85),
            )

    ax.legend(
        loc=legend_loc,
        bbox_to_anchor=(0.0, 0.72),
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        handlelength=2.0,
    )


def save_cache_npz(path, pos_sd, neg_sd, pos0, neg0, pos1, neg1, meta=None):
    if meta is None:
        meta = {}
    np.savez_compressed(
        path,
        pos_sd=pos_sd, neg_sd=neg_sd,
        pos0=pos0, neg0=neg0,
        pos1=pos1, neg1=neg1,
        **{f"meta_{k}": str(v) for k, v in meta.items()},
    )
    print("[cache] saved:", path)


def load_cache_npz(path):
    d = np.load(path, allow_pickle=True)
    print("[cache] loaded:", path)
    return d["pos_sd"], d["neg_sd"], d["pos0"], d["neg0"], d["pos1"], d["neg1"]


def split_dataset_indices(n, seed=0, frac=0.5):
    """Return two disjoint index arrays from [0..n-1]."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(frac * n)
    return idx[:cut], idx[cut:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root_dir")
    p.add_argument("--ckpt", required=False, help="Required unless using --plot_only with cache.")
    p.add_argument("--source", nargs="+", type=int, required=False)
    p.add_argument("--target", nargs="+", type=int, required=False)

    p.add_argument("--k", type=int, default=15)
    p.add_argument("--pairs", type=int, default=50000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--ft_steps", type=int, default=800)
    p.add_argument("--ft_lr", type=float, default=5e-4)
    p.add_argument("--unfreeze_last", action="store_true")

    p.add_argument("--cache", default=None)
    p.add_argument("--plot_only", action="store_true")
    p.add_argument("--out", default="cosine_gap.png")
    args = p.parse_args()

    # -------- Load or compute arrays --------
    if args.plot_only:
        if args.cache is None or not os.path.exists(args.cache):
            raise FileNotFoundError("--plot_only requires existing --cache .npz")
        pos_sd, neg_sd, pos0, neg0, pos1, neg1 = load_cache_npz(args.cache)

    else:
        if args.cache is not None and os.path.exists(args.cache):
            pos_sd, neg_sd, pos0, neg0, pos1, neg1 = load_cache_npz(args.cache)

        else:
            if args.ckpt is None:
                raise ValueError("Missing --ckpt (required for compute mode).")
            if not args.source or not args.target:
                raise ValueError("Missing --source/--target (required for compute mode).")

            set_seed(args.seed)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("[viz] device:", device)

            ckpt = torch.load(args.ckpt, map_location=device)
            emb_dim = ckpt["emb_dim"]
            num_classes = len(ckpt["mac_to_y"])

            encoder = Encoder(emb_dim=emb_dim).to(device)
            encoder.load_state_dict(ckpt["encoder"])

            classifier = Classifier(emb_dim=emb_dim, num_classes=num_classes).to(device)
            classifier.load_state_dict(ckpt["classifier"])

            # splits
            _, src_val, tgt_adapt, tgt_test, _ = make_splits(
                args.root_dir, args.source, args.target, k_per_class=args.k, seed=args.seed
            )

            # loaders
            src_loader = DataLoader(src_val, batch_size=args.batch, shuffle=False, num_workers=0)
            tgt_loader = DataLoader(tgt_test, batch_size=args.batch, shuffle=False, num_workers=0)

            # ----- (A) SAME-DOMAIN baseline: split src_val into two halves -----
            idx_a, idx_b = split_dataset_indices(len(src_val), seed=args.seed, frac=0.5)
            src_a = Subset(src_val, idx_a)
            src_b = Subset(src_val, idx_b)
            src_a_loader = DataLoader(src_a, batch_size=args.batch, shuffle=False, num_workers=0)
            src_b_loader = DataLoader(src_b, batch_size=args.batch, shuffle=False, num_workers=0)

            Za, Ya = collect_embeddings(encoder, src_a_loader, device)
            Zb, Yb = collect_embeddings(encoder, src_b_loader, device)
            pos_sd, neg_sd = sample_pairwise_cosines(Za, Ya, Zb, Yb, n_pairs=args.pairs, seed=args.seed)

            # ----- (B) CROSS-DOMAIN before adaptation -----
            Zs0, Ys0 = collect_embeddings(encoder, src_loader, device)
            Zt0, Yt0 = collect_embeddings(encoder, tgt_loader, device)
            pos0, neg0 = sample_pairwise_cosines(Zs0, Ys0, Zt0, Yt0, n_pairs=args.pairs, seed=args.seed)

            # ----- (C) CROSS-DOMAIN after few-shot adaptation -----
            adapt_loader = DataLoader(tgt_adapt, batch_size=min(64, args.batch), shuffle=True, num_workers=0)
            fewshot_finetune(
                encoder, classifier, adapt_loader, device,
                steps=args.ft_steps, lr=args.ft_lr, unfreeze_last=args.unfreeze_last
            )

            Zs1, Ys1 = collect_embeddings(encoder, src_loader, device)
            Zt1, Yt1 = collect_embeddings(encoder, tgt_loader, device)
            pos1, neg1 = sample_pairwise_cosines(Zs1, Ys1, Zt1, Yt1, n_pairs=args.pairs, seed=args.seed + 1)

            if args.cache is not None:
                meta = dict(
                    ckpt=args.ckpt, source=args.source, target=args.target,
                    k=args.k, pairs=args.pairs,
                    ft_steps=args.ft_steps, ft_lr=args.ft_lr,
                    unfreeze_last=args.unfreeze_last, seed=args.seed
                )
                save_cache_npz(args.cache, pos_sd, neg_sd, pos0, neg0, pos1, neg1, meta=meta)

    # -------- Plot 3 panels --------
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.2), sharey=True)

    plot_gap_pubready(axes[0], pos_sd, neg_sd, title="Same domain (no adapt)")
    plot_gap_pubready(axes[1], pos0, neg0, title="Cross-domain (no adapt)")
    plot_gap_pubready(axes[2], pos1, neg1, title="Cross-domain (after adapt)")

    # remove duplicate y-labels
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    fig.tight_layout(w_pad=1.2)

    out_png = args.out.replace(".png", "_3panel.png")
    out_pdf = args.out.replace(".png", "_3panel.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    print("[viz] saved:", out_png, "and", out_pdf)


if __name__ == "__main__":
    main()


# python viz_cosine_gap_v3.py Dataset/vmatrices \
#   --ckpt checkpoint.pt \
#   --source 1 2 3 \
#   --target 4 \
#   --k 15 \
#   --pairs 50000 \
#   --ft_steps 800 \
#   --ft_lr 5e-4 \
#   --unfreeze_last \
#   --out cosine_gap.png