#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from llava.model.builder import load_pretrained_model
import llava

# Speed optimization switches
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

IMAGE_TOKEN_INDEX = 151649 # for nvila

# =========================
# utils
# =========================


def get_attr(obj, path: str, default=None):
    cur = obj
    for p in path.split("."):
        if not hasattr(cur, p):
            return default
        cur = getattr(cur, p)
    return cur


def reduce_abs_keep_feat(x: torch.Tensor) -> torch.Tensor:
    """Treat the last dimension as the feature axis and average |x| over all other axes -> shape (D,)."""
    if x.dim() == 1:
        return x.abs().mean()
    reduce_dims = tuple(i for i in range(x.dim()) if i != (x.dim() - 1))
    return x.abs().mean(dim=reduce_dims)


def sample_random_pixel_values(
    B: int,
    C: int,
    H: int,
    W: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Interpret image_processor's (mean, std) as raw pixel statistics.
    Sample x_raw ~ N(mean, std^2) and return the normalized values (x_raw - mean) / std.
    """
    x_raw = torch.randn(B, C, H, W, device=device, dtype=dtype) * std.view(
        1, C, 1, 1
    ) + mean.view(1, C, 1, 1)
    x_norm = (x_raw - mean.view(1, C, 1, 1)) / std.view(1, C, 1, 1)
    return x_norm


# =========================
# hook storages
# =========================
act_in_vec: Dict[str, torch.Tensor] = {}  # (in_features,)
sens_out_vec: Dict[str, torch.Tensor] = {}  # (out_features,)


def fwd_hook_factory(name: str):
    def hook(mod: nn.Module, inp, out):
        x = inp[0]
        act_in_vec[name] = reduce_abs_keep_feat(x).detach()

    return hook


def bwd_hook_factory(name: str):
    def hook(mod: nn.Module, grad_in, grad_out):
        g = grad_out[0]
        sens_out_vec[name] = reduce_abs_keep_feat(g).detach()

    return hook


# =========================
# module collection (LLaVA 1.5)
# =========================


def collect_linear_modules_vila(
    model, n_llm_layers: int, include_vision: bool, include_lm_head: bool
) -> List[Tuple[str, nn.Linear]]:
    modules: List[Tuple[str, nn.Linear]] = []

    # (A) Projector (candidate for input boundary)
    for path in ["model.mm_projector", "mm_projector", "vision_projector"]:
        proj = get_attr(model, path, None)
        if proj is not None:
            for name, m in proj.named_modules():
                if isinstance(m, nn.Linear):
                    modules.append((f"proj.{name}", m))
            break

    # (B) LLM transformer blocks
    layers = get_attr(model, "language_model.model.layers", None) or get_attr(
        model, "llm.model.layers", None
    )
    if layers is not None:
        # n_llm_layers == -1 → use all layers
        n = len(layers) if n_llm_layers < 0 else min(n_llm_layers, len(layers))
        for i in range(n):
            blk = layers[i]
            # add attention projections if needed
            for subn in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            ]:
                m = get_attr(blk, subn, None)
                if isinstance(m, nn.Linear):
                    modules.append((f"llm.{i}.{subn}", m))
            for subn in [
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
                "mlp.fc1",
                "mlp.fc2",
            ]:
                m = get_attr(blk, subn, None)
                if isinstance(m, nn.Linear):
                    modules.append((f"llm.{i}.{subn}", m))

    # (C) Vision tower Linear (optional)
    if include_vision:
        for path in [
            "model.vision_tower.vision_tower",
            "vision_tower.vision_tower",
            "vision_tower",
        ]:
            vt = get_attr(model, path, None)
            if vt is not None:
                for name, m in vt.named_modules():
                    if isinstance(m, nn.Linear):
                        modules.append((f"vision.{name}", m))
                break

    # (D) lm_head (candidate for output boundary)
    if include_lm_head:
        lm_head = get_attr(model, "language_model.lm_head", None) or get_attr(
            model, "lm_head", None
        )
        if isinstance(lm_head, nn.Linear):
            modules.append(("lm_head", lm_head))

    # deduplicate
    seen = set()
    uniq: List[Tuple[str, nn.Linear]] = []
    for n, m in modules:
        if id(m) in seen:
            continue
        seen.add(id(m))
        uniq.append((n, m))
    return uniq


# =========================
# Online stats (CPU, Welford)
# =========================


class RunningStats:
    """
    Weighted Welford online mean/variance for matrices.
    Stores only mean and M2 on CPU (float32).
    """

    def __init__(self):
        self.count = 0.0
        self.mean = None  # CPU float32
        self.M2 = None  # CPU float32

    @torch.no_grad()
    def update(self, x: torch.Tensor, w: float = 1.0):
        """
        x: Tensor (any shape, GPU/half OK). Internally moved to CPU float32.
        w: weight (e.g., batch size)
        """
        xc = x.detach().to("cpu", dtype=torch.float32, non_blocking=True)
        if self.mean is None:
            self.mean = xc.clone()
            self.M2 = torch.zeros_like(xc)
            self.count = float(w)
            return

        total = self.count + float(w)
        delta = xc - self.mean
        self.mean.add_(delta * (float(w) / total))
        self.M2.add_((xc - self.mean) * delta * float(w))
        self.count = total

    @torch.no_grad()
    def finalize(self):
        if self.count <= 0:
            return None, None
        var = torch.clamp(self.M2 / self.count, min=0.0)
        std = torch.sqrt(var)
        return self.mean.contiguous(), std.contiguous()

    def num_samples(self) -> int:
        return int(self.count)


# =========================
# Self-generated text (BOS + random prefix → generate)
# =========================


@torch.no_grad()
def sample_self_text_ids_batch(
    model,
    tokenizer,
    batch_size,
    max_len,
    device,
    temperature=0.9,
    top_p=0.9,
    rep_penalty=1.05,
    prefix_min=1,
    prefix_max=3,
    avoid_special=True,
    seed=None,
    image_off=False,
):
    """
    Start with BOS + a random prefix, then generate the remaining tokens with the model.
    """
    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)

    # Randomize prefix length per sample
    prefix_lens = torch.randint(
        prefix_min, prefix_max + 1, (batch_size,), device=device
    )
    rand_ids_list = []
    for L in prefix_lens.tolist():
        if L <= 0:
            rand_ids_list.append(torch.empty(0, dtype=torch.long, device=device))
            continue
        if g is None:
            ids = torch.randint(0, vocab_size, (L,), device=device)
        else:
            perm = torch.randperm(vocab_size, generator=g, device=device)
            ids = perm[:L]
        rand_ids_list.append(ids)

    if (
        avoid_special
        and hasattr(tokenizer, "all_special_ids")
        and tokenizer.all_special_ids
    ):
        special = torch.tensor(tokenizer.all_special_ids, device=device)
        for i, ids in enumerate(rand_ids_list):
            if ids.numel() == 0:
                continue
            mask = torch.isin(ids, special)
            if mask.any():
                ids = ids.clone()
                ids[mask] = bos
                rand_ids_list[i] = ids

    # Build inputs: [BOS] + prefix
    inputs = []
    for ids in rand_ids_list:
        if ids.numel() == 0:
            inputs.append(torch.tensor([bos], device=device))
        else:
            inputs.append(torch.cat([torch.tensor([bos], device=device), ids], dim=0))

    max_inp = min(max(t.numel() for t in inputs), max_len)
    input_ids = torch.full((batch_size, max_inp), bos, device=device, dtype=torch.long)
    attn = torch.zeros_like(input_ids)
    for i, ids in enumerate(inputs):
        ids = ids[:max_inp]
        input_ids[i, : ids.numel()] = ids
        attn[i, : ids.numel()] = 1

    remain = max_len - input_ids.shape[1]
    if remain <= 0:
        seq = input_ids[:, :max_len]
        attn = attn[:, :max_len]
        return {"input_ids": seq, "attention_mask": attn}

    media = defaultdict(dict)
    out = model.generate(
        input_ids=input_ids,
        media=media,
        attention_mask=attn,
        max_new_tokens=remain,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rep_penalty,
        use_cache=True,
        generator=g,
    )

    seq = out[:, :max_len].clone() # since the out tensor is a inference tensor, clone() helps to convert it to a normal tensor

    if not image_off:
        seq = torch.cat([torch.full((seq.size(0), 1), IMAGE_TOKEN_INDEX, device=seq.device, dtype=seq.dtype), seq], dim=1)

    attn = torch.ones_like(seq)
    return {"input_ids": seq, "attention_mask": attn}


# =========================
# main
# =========================


def main():
    ap = argparse.ArgumentParser(
        description=(
            "LLaVA-1.5 | MC importance with self-generated text (psi ⊗ |W| ⊗ a), "
            "boundary rules, variance (GPU compute + CPU online stats)"
        )
    )
    ap.add_argument("--model-id", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model-name", default="llava-v1.5-7b")
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )

    ap.add_argument("--llm-layers", type=int, default=4)
    ap.add_argument("--include-vision", action="store_true")
    ap.add_argument("--include-lm-head", action="store_true")

    # Monte Carlo
    ap.add_argument("--K", type=int, default=128, help="Total number of text samples")
    ap.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (samples per step)"
    )
    ap.add_argument(
        "--M", type=int, default=2, help="Number of Rademacher probes per batch"
    )

    # Self-gen settings
    ap.add_argument("--text-len", type=int, default=64)
    ap.add_argument("--gen-temp", type=float, default=0.9)
    ap.add_argument("--gen-top-p", type=float, default=0.9)
    ap.add_argument("--gen-rep-penalty", type=float, default=1.05)
    ap.add_argument("--prefix-min", type=int, default=1)
    ap.add_argument("--prefix-max", type=int, default=3)
    ap.add_argument("--avoid-special", action="store_true")
    ap.add_argument("--seed", type=int, default=None)

    # Image synthesis input
    ap.add_argument(
        "--use-processor-stats",
        action="store_true",
        help="Use image_processor.image_mean/std",
    )
    ap.add_argument(
        "--image-off", action="store_true", help="Exclude image input (text-only)"
    )

    # Boundary rule: set input a=1 at projector, output psi=1 at lm_head
    ap.add_argument(
        "--disable-boundary-overwrite",
        action="store_true",
        help="Disable overwriting a/psi=1 at input/output boundaries",
    )

    ap.add_argument("--save-dir", type=str, default="importance_mats_mc_selfgen")
    ap.add_argument("--summary-topk", type=int, default=20)

    ap.add_argument(
        "--save-dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="dtype to save importance tensors (default: bf16)",
    )
    args = ap.parse_args()

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load LLaVA
    print(f"[Load] {args.model_id}")
    tokenizer, model, image_processor, context_len = llava.load(
        args.model_id, model_base=None,  
        torch_dtype=torch_dtype, device=device, model_only=False,)
    model.to(device=device, dtype=torch_dtype).eval()

    # Enable grads (required for probes)
    for p in model.parameters():
        p.requires_grad_(True)

    # Collect target modules + register hooks
    modules = collect_linear_modules_vila(
        model,
        n_llm_layers=args.llm_layers,
        include_vision=args.include_vision,
        include_lm_head=args.include_lm_head,
    )
    # print("  " + "\n  ".join([name for name, _ in modules]))
    print(f"[Collect] {len(modules)} Linear modules targeted")
    handles = []
    for name, m in modules:
        handles.append(m.register_forward_hook(fwd_hook_factory(name)))
        if hasattr(m, "register_full_backward_hook"):
            handles.append(m.register_full_backward_hook(bwd_hook_factory(name)))
        else:
            handles.append(m.register_backward_hook(bwd_hook_factory(name)))

    # Image resolution & preprocessing stats
    H = (getattr(image_processor, "crop_size", {}) or {}).get("height", 448)
    W_img = (getattr(image_processor, "crop_size", {}) or {}).get("width", 448)
    C = 3
    if args.use_processor_stats:
        im_mean = torch.tensor(
            getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073]),
            device=device,
            dtype=torch_dtype,
        )
        im_std = torch.tensor(
            getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711]),
            device=device,
            dtype=torch_dtype,
        )
        if im_mean.numel() != C or im_std.numel() != C:
            im_mean = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch_dtype)
            im_std = torch.tensor([0.25, 0.25, 0.25], device=device, dtype=torch_dtype)
    else:
        im_mean = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch_dtype)
        im_std = torch.tensor([0.25, 0.25, 0.25], device=device, dtype=torch_dtype)

    # Online stats buffers (CPU)
    layer_stats: Dict[str, RunningStats] = {}

    boundary_on = not args.disable_boundary_overwrite

    # ===== Monte Carlo loop =====
    model.zero_grad(set_to_none=True)
    total = 0
    B_cfg = max(1, int(args.batch_size))
    seq_len = int(args.text_len)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    while total < args.K:
        B_eff = min(B_cfg, args.K - total)
        total += B_eff

        # (A) Text: BOS + random prefix → self generation
        text_pack = sample_self_text_ids_batch(
            model,
            tokenizer,
            batch_size=B_eff,
            max_len=seq_len,
            device=device,
            temperature=args.gen_temp,
            top_p=args.gen_top_p,
            rep_penalty=args.gen_rep_penalty,
            prefix_min=args.prefix_min,
            prefix_max=args.prefix_max,
            avoid_special=args.avoid_special,
            seed=args.seed,
            image_off=args.image_off,
        )

        # (B) Image: synthetic or OFF
        if args.image_off:
            pixel_values = None
        else:
            pixel_values = sample_random_pixel_values(
                B=B_eff,
                C=C,
                H=H,
                W=W_img,
                mean=im_mean,
                std=im_std,
                device=device,
                dtype=torch_dtype,
            )

        for _ in range(args.M):
            act_in_vec.clear()
            sens_out_vec.clear()

            with torch.enable_grad():
                outputs = model(
                    input_ids=text_pack["input_ids"],
                    attention_mask=text_pack["attention_mask"],
                    media={"image": [image for image in pixel_values]},
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                # Scalar probe
                logits = outputs.logits[:, -1, :]  # (B_eff, V)
                xi = (
                    torch.randint_like(logits, low=0, high=2).float().mul_(2).sub_(1)
                )  # Rademacher
                S = (xi * logits).sum() / float(
                    B_eff
                )  # stabilize by averaging over batch

                model.zero_grad(set_to_none=True)
                S.backward()

            # Per-module matrix compute & accumulate online (immediately moved to CPU)
            for name, m in modules:
                W = m.weight.detach()
                W_abs = W.abs()
                out_features, in_features = W.shape

                a = act_in_vec.get(name, None)
                psi = sens_out_vec.get(name, None)

                # Boundary rules
                if boundary_on and name.startswith("proj."):
                    a = torch.ones(in_features, device=W.device, dtype=W.dtype)
                if boundary_on and (name == "lm_head" or name.endswith(".lm_head")):
                    psi = torch.ones(out_features, device=W.device, dtype=W.dtype)

                if (
                    (a is None)
                    or (psi is None)
                    or (a.numel() != in_features)
                    or (psi.numel() != out_features)
                ):
                    continue

                a = a.to(W_abs.device, dtype=W_abs.dtype)
                psi = psi.to(W_abs.device, dtype=W_abs.dtype)

                # To save VRAM, computing in half precision is fine (means/vars are converted to CPU float32)
                mat = (psi[:, None] * W_abs * a[None, :]).to(
                    dtype=W_abs.dtype
                )  # (out, in)

                if name not in layer_stats:
                    layer_stats[name] = RunningStats()
                layer_stats[name].update(mat, w=float(B_eff))

            # Optional: clear intermediate caches
            del outputs, logits, xi, S
            # torch.cuda.empty_cache()  # Uncomment only if memory is very tight

    # Compute mean/std and save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    total_runs = sum(s.num_samples() for s in layer_stats.values())
    for name, m in modules:
        if name not in layer_stats:
            continue

        mean_mat, std_mat = layer_stats[name].finalize()
        if mean_mat is None:
            continue

        W = m.weight.detach()
        Wnorm_fro = W.pow(2).sum().sqrt().float().item()
        save_dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        save_dtype = save_dtype_map[args.save_dtype]

        # After mean/std are computed, cast before saving (move to CPU to reduce file size and protect GPU memory)
        mean_to_save = mean_mat.detach().to(dtype=save_dtype, device="cpu", copy=True)
        std_to_save = std_mat.detach().to(dtype=save_dtype, device="cpu", copy=True)

        torch.save(
            {
                "name": name,
                "importance_matrix_mean": mean_to_save,  # saved dtype
                "importance_matrix_std": std_to_save,  # saved dtype
                "W_shape": W.shape,
                "W_fro": Wnorm_fro,
                "num_samples": layer_stats[name].num_samples(),
            },
            save_dir / f"{name.replace('.', '_')}.pt",
        )

        summary.append(
            (
                name,
                float(mean_mat.mean().item()),
                float(mean_mat.max().item()),
                float(std_mat.mean().item()),
                Wnorm_fro,
                layer_stats[name].num_samples(),
            )
        )

    summary.sort(key=lambda x: x[2], reverse=True)
    k = min(args.summary_topk, len(summary))
    print(
        f"=== Top-{k} layers by max(mean importance) over {total_runs} samples (K={args.K}, B={B_cfg}, M={args.M}) ==="
    )
    for n, m_mean, m_max, m_std_mean, wfro, cnt in summary[:k]:
        print(
            f"runs={cnt:4d}  max={m_max:10.4f}  mean={m_mean:10.4f}  std_mean={m_std_mean:10.4f}  ||W||_F={wfro:10.4f}  {n}"
        )

    # Remove hooks
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


if __name__ == "__main__":
    main()
