# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    select_unquantized_moe_backend,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "platform_method,expected_backend",
    [
        ("is_cuda", UnquantizedMoeBackend.TRITON),  # Default CUDA without FlashInfer
        ("is_rocm", UnquantizedMoeBackend.TRITON),  # ROCm without AITER
        ("is_cpu", UnquantizedMoeBackend.CPU),
        ("is_xpu", UnquantizedMoeBackend.XPU),
        ("is_tpu", UnquantizedMoeBackend.TPU),
        ("is_out_of_tree", UnquantizedMoeBackend.OOT),
    ],
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer",
    return_value=False,
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.rocm_aiter_ops.is_fused_moe_enabled",
    return_value=False,
)
def test_select_default_backend_by_platform(
    mock_aiter_enabled,
    mock_has_flashinfer,
    monkeypatch,
    platform_method,
    expected_backend,
):
    """Test default backend selection per platform with all optional
    accelerators (FlashInfer, AITER) disabled."""
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform"
    ) as mock_platform:
        # Set all platform checks to False
        mock_platform.is_cuda.return_value = False
        mock_platform.is_rocm.return_value = False
        mock_platform.is_cpu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_tpu.return_value = False
        mock_platform.is_out_of_tree.return_value = False

        # Set only the specified platform to True
        getattr(mock_platform, platform_method).return_value = True

        moe_config = make_dummy_moe_config()
        selected_backend = select_unquantized_moe_backend(
            moe_config=moe_config,
            use_dp=False,
            is_act_and_mul=True,
            has_bias=False,
        )

        assert selected_backend == expected_backend


@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer",
    return_value=False,
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.rocm_aiter_ops.is_fused_moe_enabled",
    return_value=True,
)
@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific backend selection test"
)
def test_select_rocm_aiter_backend(mock_aiter_enabled, mock_has_flashinfer):
    """Test ROCm backend selection when AITER is available."""
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform"
    ) as mock_platform:
        mock_platform.is_cuda.return_value = False
        mock_platform.is_rocm.return_value = True
        mock_platform.is_cpu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_tpu.return_value = False
        mock_platform.is_out_of_tree.return_value = False

        moe_config = make_dummy_moe_config()
        selected_backend = select_unquantized_moe_backend(
            moe_config=moe_config,
            use_dp=False,
        )

        assert selected_backend == UnquantizedMoeBackend.AITER


@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer",
    return_value=True,
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.is_supported_config_trtllm_bf16",
    return_value=(True, None),
)
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Only supported on NVIDIA platforms."
)
def test_select_cuda_flashinfer_trtllm_backend(
    mock_has_flashinfer, mock_is_supported_trtllm, monkeypatch
):
    """Test CUDA backend selection when FlashInfer TRTLLM is available and enabled."""
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform"
    ) as mock_platform:
        # Set as CUDA platform
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False
        mock_platform.is_cpu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_tpu.return_value = False
        mock_platform.is_out_of_tree.return_value = False

        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")

        moe_config = make_dummy_moe_config()

        selected_backend = select_unquantized_moe_backend(
            moe_config=moe_config,
            use_dp=False,
            is_act_and_mul=True,
            has_bias=False,
        )

        assert selected_backend == UnquantizedMoeBackend.FLASHINFER_TRTLLM


@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer",
    return_value=True,
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.is_supported_config_trtllm_bf16",
    return_value=(False, None),
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer_cutlass_fused_moe",
    return_value=True,
)
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Only supported on NVIDIA platforms."
)
def test_select_cuda_flashinfer_cutlass_backend(
    mock_has_flashinfer_cutlass,
    mock_is_supported_trtllm,
    mock_has_flashinfer,
    monkeypatch,
):
    """Test CUDA backend selection when FlashInfer TRTLLM is not available
    and FlashInfer CUTLASS is available."""
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform"
    ) as mock_platform:
        # Set as CUDA platform with Hopper capability
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False
        mock_platform.is_cpu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_tpu.return_value = False
        mock_platform.is_out_of_tree.return_value = False
        mock_platform.has_device_capability.return_value = True  # SM90+

        # Enable FlashInfer via env var
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")

        moe_config = make_dummy_moe_config()
        moe_config.activation = "silu"

        selected_backend = select_unquantized_moe_backend(
            moe_config=moe_config,
            use_dp=False,  # CUTLASS doesn't support DP
            is_act_and_mul=True,
            has_bias=False,
        )

        assert selected_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS


@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer",
    return_value=True,
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.is_supported_config_trtllm_bf16",
    return_value=(True, None),
)
@patch(
    "vllm.model_executor.layers.fused_moe.sonic_moe.is_sonic_moe_supported",
    return_value=True,
)
def test_pd_prefill_forces_sonic_backend(
    mock_sonic_supported,
    mock_is_supported_trtllm,
    mock_has_flashinfer,
    monkeypatch,
):
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform"
    ) as mock_platform:
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False
        mock_platform.is_cpu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_tpu.return_value = False
        mock_platform.is_out_of_tree.return_value = False
        mock_platform.has_device_capability.return_value = True

        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")
        monkeypatch.setenv(
            "VLLM_SONIC_MOE_PD_BACKEND_MODE", "prefill_sonic_decode_triton"
        )

        moe_config = make_dummy_moe_config()
        moe_config.activation = "silu"
        moe_config.kv_role = "kv_producer"

        selected_backend = select_unquantized_moe_backend(
            moe_config=moe_config,
            use_ep=False,
            use_dp=False,
            is_act_and_mul=True,
            has_bias=False,
        )

        assert selected_backend == UnquantizedMoeBackend.SONIC


@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer",
    return_value=True,
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.is_supported_config_trtllm_bf16",
    return_value=(True, None),
)
def test_pd_decode_forces_triton_backend(
    mock_is_supported_trtllm, mock_has_flashinfer, monkeypatch
):
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform"
    ) as mock_platform:
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False
        mock_platform.is_cpu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_tpu.return_value = False
        mock_platform.is_out_of_tree.return_value = False
        mock_platform.has_device_capability.return_value = True

        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")
        monkeypatch.setenv(
            "VLLM_SONIC_MOE_PD_BACKEND_MODE", "prefill_sonic_decode_triton"
        )

        moe_config = make_dummy_moe_config()
        moe_config.activation = "silu"
        moe_config.kv_role = "kv_consumer"

        selected_backend = select_unquantized_moe_backend(
            moe_config=moe_config,
            use_ep=False,
            use_dp=False,
            is_act_and_mul=True,
            has_bias=False,
        )

        assert selected_backend == UnquantizedMoeBackend.TRITON
