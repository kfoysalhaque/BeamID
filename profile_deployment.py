import time
import torch

from model import Encoder, Classifier
from datagen import make_splits

# ---------------------------
# FLOPs estimator (Conv1d + Linear only, matches your model.py)
# ---------------------------
def conv1d_flops(L_out, C_in, C_out, k, batch=1):
    macs = batch * L_out * C_out * (C_in * k)
    return 2 * macs  # mul+add

def linear_flops(in_f, out_f, batch=1):
    macs = batch * in_f * out_f
    return 2 * macs

def estimate_infer_flops(batch, num_classes, emb_dim=128, L=250):
    flops = 0
    # Encoder convs: (16->32,k7), (32->64,k5), (64->128,k3), length preserved
    flops += conv1d_flops(L, 16, 32, 7, batch)
    flops += conv1d_flops(L, 32, 64, 5, batch)
    flops += conv1d_flops(L, 64, 128, 3, batch)
    # Encoder fc: 128->emb_dim
    flops += linear_flops(128, emb_dim, batch)
    # Classifier: emb_dim->C
    flops += linear_flops(emb_dim, num_classes, batch)
    return flops

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def bytes_of_params(m, dtype=torch.float32):
    b = torch.tensor([], dtype=dtype).element_size()
    return count_params(m) * b

# ---------------------------
# Timing helpers
# ---------------------------
@torch.no_grad()
def bench_inference(encoder, classifier, device, batch=1, iters=200, warmup=50):
    encoder.eval(); classifier.eval()

    x = torch.randn(batch, 16, 250, device=device)
    # warmup
    for _ in range(warmup):
        z = encoder(x)
        _ = classifier(z)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        z = encoder(x)
        _ = classifier(z)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) * 1e3 / iters
    return ms

def bench_adaptation_step_time(encoder, classifier, device, batch=64, iters=200, warmup=30, unfreeze_last=False):
    # mimic adapt_eval.fewshot_finetune, but using synthetic data
    for p in encoder.parameters():
        p.requires_grad = False
    for p in classifier.parameters():
        p.requires_grad = True
    if unfreeze_last:
        for p in encoder.fc.parameters():
            p.requires_grad = True

    params = list(classifier.parameters())
    if unfreeze_last:
        params += list(encoder.fc.parameters())

    opt = torch.optim.Adam(params, lr=1e-3)

    encoder.train(); classifier.train()
    x = torch.randn(batch, 16, 250, device=device)
    y = torch.randint(0, classifier.fc.out_features, (batch,), device=device)

    # warmup
    for _ in range(warmup):
        z = encoder(x)
        logits = classifier(z)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        z = encoder(x)
        logits = classifier(z)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1e3 / iters  # ms/step

def peak_mem_mb(device):
    if not device.startswith("cuda"):
        return None
    return torch.cuda.max_memory_allocated() / (1024**2)

# ---------------------------
# Main
# ---------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("root_dir", help="path to your .npy V-matrix files")
    p.add_argument("--ckpt", default="checkpoint.pt")
    p.add_argument("--source", nargs="+", type=int, default=[1,2,3])
    p.add_argument("--target", nargs="+", type=int, default=[4,5])
    p.add_argument("--k", type=int, default=55)
    p.add_argument("--ft_steps", type=int, default=4000)
    p.add_argument("--unfreeze_last", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[profile] device:", device)

    ckpt = torch.load(args.ckpt, map_location=device)
    emb_dim = ckpt["emb_dim"]
    mac_to_y = ckpt["mac_to_y"]
    num_classes = len(mac_to_y)

    encoder = Encoder(emb_dim=emb_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])

    classifier = Classifier(emb_dim=emb_dim, num_classes=num_classes).to(device)
    if "classifier" in ckpt:
        classifier.load_state_dict(ckpt["classifier"])

    # ---- Static summaries
    print("\n=== STATIC ===")
    print("encoder params:", count_params(encoder))
    print("classifier params:", count_params(classifier))
    print("encoder size (FP32): %.2f KB" % (bytes_of_params(encoder)/1024))
    print("classifier size (FP32): %.2f KB" % (bytes_of_params(classifier)/1024))

    flops1 = estimate_infer_flops(batch=1, num_classes=num_classes, emb_dim=emb_dim)
    flops64 = estimate_infer_flops(batch=64, num_classes=num_classes, emb_dim=emb_dim)
    print("inference FLOPs/sample (B=1): %.2f MFLOPs" % (flops1/1e6))
    print("inference FLOPs/batch (B=64): %.2f MFLOPs" % (flops64/1e6))

    # ---- Runtime inference benchmark
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    print("\n=== RUNTIME: inference ===")
    ms_b1 = bench_inference(encoder, classifier, device, batch=1)
    ms_b64 = bench_inference(encoder, classifier, device, batch=64)
    print("latency (batch=1): %.3f ms" % ms_b1)
    print("latency (batch=64): %.3f ms  |  throughput: %.1f samples/s"
          % (ms_b64, 64.0 / (ms_b64/1e3)))

    if device.startswith("cuda"):
        print("peak GPU mem after inference benches: %.2f MB" % peak_mem_mb(device))

    # ---- Runtime adaptation-step benchmark (synthetic)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    print("\n=== RUNTIME: adaptation ===")
    ms_step = bench_adaptation_step_time(
        encoder, classifier, device, batch=64, unfreeze_last=args.unfreeze_last
    )
    print("adapt step time (batch=64): %.3f ms/step" % ms_step)
    total_s = (ms_step * args.ft_steps) / 1e3
    print("estimated wall time for ft_steps=%d: %.2f s" % (args.ft_steps, total_s))

    if device.startswith("cuda"):
        print("peak GPU mem during adaptation bench: %.2f MB" % peak_mem_mb(device))

    print("\nTip: for true end-to-end (real I/O + shuffling), time adapt_eval.py run.")

if __name__ == "__main__":
    main()
