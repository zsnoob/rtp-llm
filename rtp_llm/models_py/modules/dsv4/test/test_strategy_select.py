"""Unit test for ``dsv4/moe/strategies/base.py::select_strategy``.

Covers the priority matrix in the strategy module docstring + ``forced``
override + legacy env-toggle resolution + the explicit-fail-on-mismatch
contract. Pure-Python, no CUDA / DeepGEMM / dist required — runs on host.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock

import torch

# Importing strategies populates the registry via ``register_strategy``.
from rtp_llm.models_py.modules.dsv4.moe import mega_se_buf
from rtp_llm.models_py.modules.dsv4.moe.strategies import (
    DeepEPStrategy,
    GroupedFP4Strategy,
    LocalLoopStrategy,
    MegaMoEStrategy,
    MegaMoEStrategySE,
    MoeCfg,
    _has_fp8_fp4_grouped_kernel,
    select_strategy,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies.base import _resolve_forced


def _cfg(ep_size: int = 1) -> MoeCfg:
    """A minimal MoeCfg sufficient for ``can_handle`` checks."""
    n_local = 256 // max(ep_size, 1)
    return MoeCfg(
        layer_id=2,
        dim=7168,
        moe_inter_dim=2048,
        n_routed_experts=256,
        n_activated_experts=6,
        swiglu_limit=10.0,
        ep_size=ep_size,
        ep_rank=0,
        n_local_experts=n_local,
        local_expert_start=0,
        local_expert_end=n_local,
        max_tokens_per_rank=8192,
    )


@contextmanager
def _env(**kw):
    """Temporarily set env vars; ``None`` value pops the var."""
    saved = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class StrategySelectTest(unittest.TestCase):
    """Cover the (ep_size, kernel_avail, mega_avail) matrix."""

    def setUp(self):
        # Ensure clean env baseline for every test.
        for k in (
            "DSV4_MOE_STRATEGY",
            "DSV4_USE_MEGA_MOE",
            "DSV4_USE_MEGA_MOE_SE",
            "DSV4_USE_MEGA_MOE_FUSED",
            "DSV4_USE_GROUPED_FP4",
        ):
            os.environ.pop(k, None)

    # --- auto-pick matrix --------------------------------------------------

    def test_ep1_with_grouped_kernel_picks_grouped(self):
        with mock.patch.object(
            GroupedFP4Strategy, "can_handle", return_value=True
        ), mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False):
            self.assertIs(select_strategy(_cfg(ep_size=1)), GroupedFP4Strategy)

    def test_grouped_selection_is_gated_by_ep_size(self):
        cfg = _cfg(ep_size=2)
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "_has_fp8_fp4_grouped_kernel",
            return_value=True,
        ):
            self.assertFalse(GroupedFP4Strategy.can_handle(cfg))

    def test_grouped_kernel_probe_requires_sm100(self):
        fake_deep_gemm = types.SimpleNamespace(
            m_grouped_fp8_fp4_gemm_nt_contiguous=object(),
            get_mk_alignment_for_contiguous_layout=lambda: (128, 128),
        )
        with mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.get_device_capability",
            return_value=(12, 0),
        ):
            self.assertFalse(_has_fp8_fp4_grouped_kernel())

        with mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.get_device_capability",
            return_value=(10, 0),
        ):
            self.assertTrue(_has_fp8_fp4_grouped_kernel())

    def test_ep1_no_grouped_falls_to_local(self):
        with mock.patch.object(
            GroupedFP4Strategy, "can_handle", return_value=False
        ), mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=False
        ), mock.patch.object(
            DeepEPStrategy, "can_handle", return_value=False
        ):
            self.assertIs(select_strategy(_cfg(ep_size=1)), LocalLoopStrategy)

    def test_ep_gt1_with_mega_picks_mega(self):
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=True):
            self.assertIs(select_strategy(_cfg(ep_size=4)), MegaMoEStrategy)

    def test_ep_gt1_default_stays_mega_when_se_is_capable(self):
        with mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=True
        ), mock.patch.object(MegaMoEStrategySE, "can_handle", return_value=True):
            self.assertIs(select_strategy(_cfg(ep_size=4)), MegaMoEStrategy)

    def test_ep_gt1_no_mega_raises(self):
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=4))
        self.assertIn("requires MegaMoEStrategy", str(cm.exception))
        self.assertIn("fallback to DeepEP/LocalLoop is disabled", str(cm.exception))

    # --- forced override ---------------------------------------------------

    def test_forced_known_and_capable_returns_it(self):
        self.assertIs(
            select_strategy(_cfg(ep_size=1), forced="local_loop"),
            LocalLoopStrategy,
        )

    def test_forced_known_but_incapable_raises(self):
        # Force grouped_fp4 with grouped kernel mocked unavailable.
        with mock.patch.object(GroupedFP4Strategy, "can_handle", return_value=False):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=1), forced="grouped_fp4")
        self.assertIn("Forced MoE strategy 'grouped_fp4'", str(cm.exception))
        self.assertIn("cannot handle", str(cm.exception))

    def test_forced_ep_gt1_non_mega_raises_even_if_capable(self):
        with mock.patch.object(DeepEPStrategy, "can_handle", return_value=True):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=4), forced="deepep")
        self.assertIn("requires MegaMoEStrategy", str(cm.exception))
        self.assertIn("bypass Mega", str(cm.exception))

    def test_forced_unknown_raises(self):
        with self.assertRaises(RuntimeError) as cm:
            select_strategy(_cfg(), forced="bogus")
        self.assertIn("Unknown MoE strategy 'bogus'", str(cm.exception))
        self.assertIn("Available", str(cm.exception))

    # --- env resolution ----------------------------------------------------

    def test_env_dsv4_moe_strategy_overrides_ctor(self):
        with _env(DSV4_MOE_STRATEGY="local_loop"):
            self.assertEqual(_resolve_forced(None), ("local_loop", True))
            self.assertEqual(_resolve_forced("mega"), ("local_loop", True))

    def test_env_dsv4_moe_strategy_auto_falls_through(self):
        with _env(DSV4_MOE_STRATEGY="auto"):
            self.assertEqual(_resolve_forced(None), (None, False))
            self.assertEqual(_resolve_forced("mega"), ("mega", True))

    def test_legacy_use_mega_moe_1_translates_to_mega_nonstrict(self):
        # Legacy toggle is non-strict: ``select_strategy`` falls through to
        # auto-pick when the named strategy can't handle the cfg (e.g.
        # ep_size=1 + Mega). Smokes commonly leave DSV4_USE_MEGA_MOE=1
        # ON across configs that include ep_size=1.
        with _env(DSV4_USE_MEGA_MOE="1"):
            self.assertEqual(_resolve_forced(None), ("mega", False))

    def test_mega_moe_se_opt_in_is_strict(self):
        with _env(DSV4_USE_MEGA_MOE_SE="1"):
            self.assertEqual(_resolve_forced(None), ("mega_se", True))

    def test_mega_moe_se_opt_in_accepts_generic_mega_hint(self):
        with _env(DSV4_USE_MEGA_MOE_SE="1", DSV4_USE_MEGA_MOE="1"):
            self.assertEqual(_resolve_forced(None), ("mega_se", True))

    def test_mega_moe_se_opt_in_accepts_generic_mega_ctor(self):
        with _env(DSV4_USE_MEGA_MOE_SE="1"):
            self.assertEqual(_resolve_forced("mega"), ("mega_se", True))

    def test_mega_moe_se_and_grouped_conflict(self):
        with _env(
            DSV4_USE_MEGA_MOE_SE="1",
            DSV4_USE_GROUPED_FP4="1",
        ):
            with self.assertRaises(RuntimeError) as cm:
                _resolve_forced(None)
        self.assertIn("Conflicting", str(cm.exception))

    def test_mega_moe_se_opt_in_selects_se(self):
        with _env(DSV4_USE_MEGA_MOE_SE="1"), mock.patch.object(
            MegaMoEStrategySE, "can_handle", return_value=True
        ):
            forced, strict = _resolve_forced(None)
            self.assertIs(
                select_strategy(_cfg(ep_size=2), forced=forced, strict=strict),
                MegaMoEStrategySE,
            )

    def test_mega_moe_se_unavailable_fails_loudly(self):
        with _env(DSV4_USE_MEGA_MOE_SE="1"), mock.patch.object(
            MegaMoEStrategySE, "can_handle", return_value=False
        ):
            forced, strict = _resolve_forced(None)
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=2), forced=forced, strict=strict)
        self.assertIn("Forced MoE strategy 'mega_se'", str(cm.exception))

    def test_official_deepgemm_mega_se_signature_is_accepted(self):
        def fp8_fp4_mega_moe(
            y,
            l1_weights,
            l2_weights,
            sym_buffer,
            shared_l1_weights=None,
            shared_l2_weights=None,
            cumulative_local_expert_recv_stats=None,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=None,
            fast_math=True,
        ):
            pass

        def get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            num_shared_experts=0,
            use_fp8_dispatch=True,
            mma_type="fp8xfp4",
            activation="swiglu",
        ):
            pass

        fake_deep_gemm = types.SimpleNamespace(
            fp8_fp4_mega_moe=fp8_fp4_mega_moe,
            get_symm_buffer_for_mega_moe=get_symm_buffer_for_mega_moe,
            get_block_m_for_mega_moe=object(),
            transform_weights_for_mega_moe=object(),
            transform_sf_into_required_layout=object(),
        )
        with _env(DSV4_USE_MEGA_MOE="1"), mock.patch.object(
            mega_se_buf, "_mega_moe_unavailable_reason", return_value=None
        ), mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}):
            self.assertIsNone(mega_se_buf._mega_moe_se_unavailable_reason())

    def test_mega_moe_se_launch_matches_official_api(self):
        calls = []

        fake_deep_gemm = types.SimpleNamespace(
            fp8_fp4_mega_moe=lambda *args, **kwargs: calls.append((args, kwargs))
        )
        strategy = object.__new__(MegaMoEStrategySE)
        strategy.cfg = types.SimpleNamespace(layer_id=2, swiglu_limit=10.0)
        strategy._mega_l1_w = "routed_l1_w"
        strategy._mega_l1_sf = "routed_l1_sf"
        strategy._mega_l2_w = "routed_l2_w"
        strategy._mega_l2_sf = "routed_l2_sf"
        strategy._se_l1_w = "shared_l1_w"
        strategy._se_l1_sf = "shared_l1_sf"
        strategy._se_l2_w = "shared_l2_w"
        strategy._se_l2_sf = "shared_l2_sf"
        strategy._mega_buf = "symm_buffer"

        with mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}), mock.patch.object(
            strategy, "_maybe_pre_kernel_barrier"
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.mega_se."
            "sync_cuda_graph_warmup_ranks"
        ):
            strategy._launch("output", 4, "cuda:0")

        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(args[0], "output")
        self.assertEqual(kwargs["shared_l1_weights"], ("shared_l1_w", "shared_l1_sf"))
        self.assertEqual(kwargs["shared_l2_weights"], ("shared_l2_w", "shared_l2_sf"))
        self.assertNotIn("shared_recipe", kwargs)

    def test_mega_moe_se_expands_checkpoint_scale_without_requantizing(self):
        transformed = []

        fake_deep_gemm = types.SimpleNamespace(
            transform_sf_into_required_layout=lambda *args, **kwargs: transformed.append(
                (args, kwargs)
            )
            or "packed_scale"
        )
        scale = torch.tensor(
            [[1.0, 2.0], [4.0, 8.0]], dtype=torch.float8_e8m0fnu
        )

        result = MegaMoEStrategySE._shared_expert_sf_to_int(
            fake_deep_gemm, scale, 256, 256
        )

        self.assertEqual(result, "packed_scale")
        self.assertEqual(len(transformed), 1)
        args, kwargs = transformed[0]
        expanded, mn, k, recipe = args
        self.assertEqual((mn, k, recipe), (256, 256, (1, 32)))
        self.assertEqual(kwargs, {"num_groups": None})
        self.assertEqual(tuple(expanded.shape), (256, 8))
        torch.testing.assert_close(
            expanded[0], torch.tensor([1.0] * 4 + [2.0] * 4)
        )
        torch.testing.assert_close(
            expanded[128], torch.tensor([4.0] * 4 + [8.0] * 4)
        )

    def test_mega_moe_se_and_old_fused_conflict(self):
        with _env(
            DSV4_USE_MEGA_MOE_SE="1",
            DSV4_USE_MEGA_MOE_FUSED="1",
        ):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=2))
        self.assertIn("select exactly one Mega variant", str(cm.exception))

    def test_legacy_use_grouped_fp4_1_translates_to_grouped_nonstrict(self):
        with _env(DSV4_USE_GROUPED_FP4="1"):
            self.assertEqual(_resolve_forced(None), ("grouped_fp4", False))

    def test_legacy_conflicting_positives_raise(self):
        with _env(DSV4_USE_MEGA_MOE="1", DSV4_USE_GROUPED_FP4="1"):
            with self.assertRaises(RuntimeError) as cm:
                _resolve_forced(None)
            self.assertIn("Conflicting", str(cm.exception))

    def test_legacy_conflicting_with_ctor_raises(self):
        with _env(DSV4_USE_MEGA_MOE="1"):
            with self.assertRaises(RuntimeError) as cm:
                _resolve_forced("grouped_fp4")
            self.assertIn("Conflicting MoE strategy", str(cm.exception))

    def test_legacy_negation_does_not_force_alternative(self):
        # DSV4_USE_MEGA_MOE=0 should NOT force a different strategy. EP>1
        # select_strategy() treats disabled Mega as a fatal config error.
        with _env(DSV4_USE_MEGA_MOE="0"):
            self.assertEqual(_resolve_forced(None), (None, False))

    def test_legacy_negation_ep_gt1_raises(self):
        with _env(DSV4_USE_MEGA_MOE="0"):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=4))
        self.assertIn("DSV4_USE_MEGA_MOE=0 disables Mega MoE", str(cm.exception))

    def test_legacy_force_nonstrict_falls_through_when_incapable(self):
        # Legacy DSV4_USE_MEGA_MOE=1 + ep_size=1 cfg: Mega.can_handle False
        # because ep_size=1; should silently fall through to LocalLoop
        # (NOT raise — that's the strict-mode behaviour). Mirrors the
        # 64k_cp4_ep1 smoke that has ep_size=1 + DSV4_USE_MEGA_MOE=1.
        with mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=False
        ), mock.patch.object(GroupedFP4Strategy, "can_handle", return_value=False):
            self.assertIs(
                select_strategy(_cfg(ep_size=1), forced="mega", strict=False),
                LocalLoopStrategy,
            )


if __name__ == "__main__":
    unittest.main()
