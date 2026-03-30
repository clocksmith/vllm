# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sweep MoE backend strategies across shapes and emit JSON results.

This benchmark focuses on CUDA unquantized MoE and compares:
- always_triton
- always_sonic (when Sonic is valid for the shape/config)
- pd_split (prefill: Sonic if valid else Triton, decode: Triton)

It also records what the runtime auto-selector would choose for each phase
under current environment settings (e.g., FlashInfer/Sonic toggles).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch

from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts,
    get_config_file_name,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEModularKernel
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    select_unquantized_moe_backend,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.fused_moe.sonic_moe import (
    SonicMoeExperts,
    is_sonic_moe_supported,
    is_valid_sonic_moe,
    permute_weights_for_sonic,
)
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    is_workspace_manager_initialized,
)

DEFAULT_PREFILL_M = [256, 512, 1024]
DEFAULT_DECODE_M = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_HIDDEN_K = [512, 1024, 2048]
DEFAULT_INTERMEDIATE_I = [1024, 2048, 4096]  # N = 2 * I
DEFAULT_EXPERTS_E = [8, 16, 32, 64]
DEFAULT_TOPK = [1, 2, 4, 8, 16, 32]


def _parse_int_list(v: str) -> list[int]:
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def _bench_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def _weight_bytes(e: int, n: int, k: int, dtype: torch.dtype) -> int:
    b = torch.tensor([], dtype=dtype).element_size()
    w1 = e * n * k * b
    w2 = e * k * (n // 2) * b
    return w1 + w2


def _has_tuned_triton_config(num_experts: int, n_half: int) -> bool:
    from vllm.model_executor.layers.fused_moe import fused_moe as fm

    cfg_file = get_config_file_name(num_experts, n_half, dtype=None)
    cfg_path = (
        Path(os.path.dirname(os.path.realpath(fm.__file__))) / "configs" / cfg_file
    )
    return cfg_path.exists()


def _sonic_invalid_reason(topk: int, dtype: torch.dtype, k: int, n: int) -> str:
    if topk > 16:
        return "topk_gt_16"
    if dtype not in (torch.float16, torch.bfloat16):
        return "dtype_unsupported"
    if k < 512 or (k % 64) != 0:
        return "hidden_dim_constraint"
    if (n // 2) % 64 != 0:
        return "intermediate_dim_constraint"
    return "shape_or_runtime_constraint"


def _make_moe_config(
    *,
    e: int,
    topk: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    kv_role: str,
) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=e,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size_per_partition=n // 2,
        num_local_experts=e,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation="silu",
        in_dtype=dtype,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        kv_role=kv_role,
    )


def _run_one_case(
    *,
    phase: str,
    m: int,
    k: int,
    n: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    auto_use_ep: bool,
    auto_use_dp: bool,
) -> dict[str, Any]:
    hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n // 2), device="cuda", dtype=dtype) / 10

    router_logits = torch.randn((m, e), device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1).to(dtype)

    kv_role = "kv_producer" if phase == "prefill" else "kv_consumer"
    moe_config = _make_moe_config(
        e=e, topk=topk, k=k, n=n, dtype=dtype, kv_role=kv_role
    )

    auto_backend = select_unquantized_moe_backend(
        moe_config=moe_config,
        use_ep=auto_use_ep,
        use_dp=auto_use_dp,
        is_act_and_mul=True,
        has_bias=False,
    ).name

    triton_kernel = FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(),
        TritonExperts(moe_config, FUSED_MOE_UNQUANTIZED_CONFIG),
        inplace=False,
    )
    run_triton = partial(
        triton_kernel,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    triton_us: float | None = None
    sonic_us: float | None = None
    rel_err: float | None = None
    sonic_valid = False
    sonic_reason = ""

    with torch.inference_mode():
        # Triton is always benchmarked.
        triton_us = _bench_us(run_triton, warmup, iters)
        out_triton = run_triton()

        sonic_valid = is_valid_sonic_moe(hidden_states, w1, w2, e, topk)
        if sonic_valid:
            w1_sonic = permute_weights_for_sonic(w1)
            sonic_kernel = FusedMoEModularKernel(
                MoEPrepareAndFinalizeNoEP(),
                SonicMoeExperts(
                    moe_config,
                    FUSED_MOE_UNQUANTIZED_CONFIG,
                    weights_prepermuted=True,
                ),
                inplace=False,
            )
            run_sonic = partial(
                sonic_kernel,
                hidden_states=hidden_states,
                w1=w1_sonic,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )
            out_sonic = run_sonic()
            denom = out_triton.abs().max().clamp_min(1e-6)
            rel_err = ((out_sonic - out_triton).abs().max() / denom).item()
            sonic_us = _bench_us(run_sonic, warmup, iters)
        else:
            sonic_reason = _sonic_invalid_reason(topk, dtype, k, n)

    pd_split_us = triton_us
    if phase == "prefill" and sonic_us is not None:
        pd_split_us = sonic_us

    strategy_latencies: dict[str, float] = {
        "always_triton": triton_us,
        "pd_split": pd_split_us,
    }
    if sonic_us is not None:
        strategy_latencies["always_sonic"] = sonic_us

    best_strategy = min(strategy_latencies, key=strategy_latencies.get)
    best_us = strategy_latencies[best_strategy]

    return {
        "phase": phase,
        "dtype": "bf16" if dtype == torch.bfloat16 else "fp16",
        "m": m,
        "k": k,
        "n": n,
        "e": e,
        "topk": topk,
        "weight_gb": round(_weight_bytes(e, n, k, dtype) / (1024**3), 4),
        "triton_us": triton_us,
        "sonic_us": sonic_us,
        "pd_split_us": pd_split_us,
        "sonic_speedup_vs_triton": (triton_us / sonic_us) if sonic_us else None,
        "rel_err_sonic_vs_triton": rel_err,
        "sonic_valid": sonic_valid,
        "sonic_invalid_reason": sonic_reason,
        "auto_backend": auto_backend,
        "best_strategy": best_strategy,
        "best_us": best_us,
    }


def _default_output_path() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"benchmark_moe_backend_sweep_{ts}.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep MoE backend strategies (Triton/Sonic/PD split) and emit JSON."
        )
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "both"], default="both")
    parser.add_argument(
        "--prefill-m", type=str, default=",".join(map(str, DEFAULT_PREFILL_M))
    )
    parser.add_argument(
        "--decode-m", type=str, default=",".join(map(str, DEFAULT_DECODE_M))
    )
    parser.add_argument(
        "--k-list", type=str, default=",".join(map(str, DEFAULT_HIDDEN_K))
    )
    parser.add_argument(
        "--i-list", type=str, default=",".join(map(str, DEFAULT_INTERMEDIATE_I))
    )
    parser.add_argument(
        "--e-list", type=str, default=",".join(map(str, DEFAULT_EXPERTS_E))
    )
    parser.add_argument(
        "--topk-list", type=str, default=",".join(map(str, DEFAULT_TOPK))
    )
    parser.add_argument(
        "--max-weight-gb",
        type=float,
        default=1.5,
        help="Skip shapes whose MoE weights exceed this per-layer size.",
    )
    parser.add_argument(
        "--require-tuned-triton",
        action="store_true",
        help="Only keep shapes with a tuned Triton MoE config file for this device.",
    )
    parser.add_argument(
        "--auto-use-ep",
        action="store_true",
        help="Pass use_ep=True when querying auto backend selector (metadata only).",
    )
    parser.add_argument(
        "--auto-use-dp",
        action="store_true",
        help="Pass use_dp=True when querying auto backend selector (metadata only).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Path to write JSON results. Defaults to a timestamped filename.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        return 0

    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda"))

    prefill_m = _parse_int_list(args.prefill_m)
    decode_m = _parse_int_list(args.decode_m)
    k_list = _parse_int_list(args.k_list)
    i_list = _parse_int_list(args.i_list)
    e_list = _parse_int_list(args.e_list)
    topk_list = _parse_int_list(args.topk_list)
    n_list = [2 * i for i in i_list]

    dtypes = (
        [torch.bfloat16, torch.float16]
        if args.dtype == "both"
        else [torch.bfloat16 if args.dtype == "bf16" else torch.float16]
    )

    all_shapes: list[tuple[str, int, int, int, int, int, torch.dtype]] = []
    for phase, m_values in (("prefill", prefill_m), ("decode", decode_m)):
        for m in m_values:
            for k in k_list:
                for n in n_list:
                    for e in e_list:
                        for topk in topk_list:
                            for dtype in dtypes:
                                all_shapes.append((phase, m, k, n, e, topk, dtype))

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    sonic_available = is_sonic_moe_supported()

    for phase, m, k, n, e, topk, dtype in all_shapes:
        weight_gb = _weight_bytes(e, n, k, dtype) / (1024**3)
        if weight_gb > args.max_weight_gb:
            skipped.append(
                {
                    "phase": phase,
                    "m": m,
                    "k": k,
                    "n": n,
                    "e": e,
                    "topk": topk,
                    "dtype": "bf16" if dtype == torch.bfloat16 else "fp16",
                    "reason": "weight_gb_limit",
                    "weight_gb": round(weight_gb, 4),
                }
            )
            continue
        if args.require_tuned_triton and not _has_tuned_triton_config(e, n // 2):
            skipped.append(
                {
                    "phase": phase,
                    "m": m,
                    "k": k,
                    "n": n,
                    "e": e,
                    "topk": topk,
                    "dtype": "bf16" if dtype == torch.bfloat16 else "fp16",
                    "reason": "missing_tuned_triton_config",
                }
            )
            continue
        try:
            row = _run_one_case(
                phase=phase,
                m=m,
                k=k,
                n=n,
                e=e,
                topk=topk,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                auto_use_ep=args.auto_use_ep,
                auto_use_dp=args.auto_use_dp,
            )
            results.append(row)
            print(
                f"[ok] phase={phase} dtype={row['dtype']} m={m} k={k} n={n} "
                f"e={e} topk={topk} triton={row['triton_us']:.2f}us "
                f"sonic={format(row['sonic_us'], '.2f') if row['sonic_us'] else 'n/a'} "
                f"best={row['best_strategy']}"
            )
        except RuntimeError as exc:
            skipped.append(
                {
                    "phase": phase,
                    "m": m,
                    "k": k,
                    "n": n,
                    "e": e,
                    "topk": topk,
                    "dtype": "bf16" if dtype == torch.bfloat16 else "fp16",
                    "reason": "runtime_error",
                    "error": str(exc),
                }
            )

    payload = {
        "meta": {
            "device": torch.cuda.get_device_name(0),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sonic_available": sonic_available,
            "args": vars(args),
            "num_results": len(results),
            "num_skipped": len(skipped),
        },
        "results": results,
        "skipped": skipped,
    }

    output_path = args.output_json or _default_output_path()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote JSON: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
