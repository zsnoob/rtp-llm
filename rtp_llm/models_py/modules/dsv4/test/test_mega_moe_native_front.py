from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from rtp_llm.models_py.modules.dsv4.block import Block
from rtp_llm.models_py.modules.dsv4.moe.native_front import (
    FLASH_MOE_FRONT_GEOMETRY,
    MOE_FRONT_EXPERTS,
    MOE_FRONT_HC_MULT,
    MOE_FRONT_HC_WIDTH,
    MOE_FRONT_HIDDEN,
    MOE_FRONT_KERNEL_CONTRACT_VERSION,
    MOE_FRONT_MAX_M,
    MOE_FRONT_SCALE_COLS,
    MOE_FRONT_TOPK,
    PRO_MOE_FRONT_GEOMETRY,
    MegaMoEFrontAdapter,
    MegaMoEFrontRuntime,
)
from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer
from rtp_llm.utils.model_weight import W


def _geometry(spec=PRO_MOE_FRONT_GEOMETRY, **overrides):
    geometry = {
        "abi_version": 1,
        "kernel_contract_version": MOE_FRONT_KERNEL_CONTRACT_VERSION,
        "hidden": spec.hidden,
        "hc_mult": spec.hc_mult,
        "hc_width": spec.hc_width,
        "experts": spec.experts,
        "topk": spec.topk,
        "max_m": spec.max_m,
        "scale_cols": spec.scale_cols,
        "collapse_ssq_bits": 32,
    }
    geometry.update(overrides)
    return geometry


class _FakeOps:
    def __init__(self, geometry=None) -> None:
        self.geometry = _geometry() if geometry is None else geometry
        self.plan_calls = []
        self.learned_calls = []
        self.hash_calls = []
        owner = self

        class Dsv4MoeFrontPlan:
            def __init__(self, hidden_states, hc_fn, logical_m):
                self.hidden_states = hidden_states
                self.hc_fn = hc_fn
                self.logical_m = logical_m
                owner.plan_calls.append(self)

            def run_learned_out(
                self,
                *,
                hc_base,
                hc_scale,
                ffn_norm_weight,
                router_weight,
                correction_bias,
                collapsed,
                collapse_ssq,
                normalized_mix,
                normalized,
                x_fp8,
                x_sf,
                shared_l1_x_sf,
                topk_ids,
                topk_weights,
                post,
                comb,
                shared_block_m,
                router_logits,
                norm_eps,
                hc_eps,
                route_scale,
                use_pdl,
            ):
                owner.learned_calls.append(
                    {
                        "plan": self,
                        "hc_base": hc_base,
                        "hc_scale": hc_scale,
                        "ffn_norm_weight": ffn_norm_weight,
                        "router_weight": router_weight,
                        "correction_bias": correction_bias,
                        "collapsed": collapsed,
                        "collapse_ssq": collapse_ssq,
                        "normalized_mix": normalized_mix,
                        "normalized": normalized,
                        "x_fp8": x_fp8,
                        "x_sf": x_sf,
                        "shared_l1_x_sf": shared_l1_x_sf,
                        "topk_ids": topk_ids,
                        "topk_weights": topk_weights,
                        "post": post,
                        "comb": comb,
                        "shared_block_m": shared_block_m,
                        "router_logits": router_logits,
                        "norm_eps": norm_eps,
                        "hc_eps": hc_eps,
                        "route_scale": route_scale,
                        "use_pdl": use_pdl,
                    }
                )

            def run_hash_out(
                self,
                *,
                hc_base,
                hc_scale,
                ffn_norm_weight,
                router_weight,
                input_ids,
                tid2eid,
                collapsed,
                collapse_ssq,
                normalized_mix,
                normalized,
                x_fp8,
                x_sf,
                shared_l1_x_sf,
                router_logits,
                topk_ids,
                topk_weights,
                post,
                comb,
                shared_block_m,
                norm_eps,
                hc_eps,
                route_scale,
                use_pdl,
            ):
                owner.hash_calls.append(
                    {
                        "plan": self,
                        "hc_base": hc_base,
                        "hc_scale": hc_scale,
                        "ffn_norm_weight": ffn_norm_weight,
                        "router_weight": router_weight,
                        "input_ids": input_ids,
                        "tid2eid": tid2eid,
                        "collapsed": collapsed,
                        "collapse_ssq": collapse_ssq,
                        "normalized_mix": normalized_mix,
                        "normalized": normalized,
                        "x_fp8": x_fp8,
                        "x_sf": x_sf,
                        "shared_l1_x_sf": shared_l1_x_sf,
                        "router_logits": router_logits,
                        "topk_ids": topk_ids,
                        "topk_weights": topk_weights,
                        "post": post,
                        "comb": comb,
                        "shared_block_m": shared_block_m,
                        "norm_eps": norm_eps,
                        "hc_eps": hc_eps,
                        "route_scale": route_scale,
                        "use_pdl": use_pdl,
                    }
                )

        self.Dsv4MoeFrontPlan = Dsv4MoeFrontPlan

    def geometry_moe_front(self, hidden=MOE_FRONT_HIDDEN):
        return self.geometry

    def build_info_moe_front(self):
        return {
            "source_commit": "c81d23db",
            "source_sha256": "1" * 64,
            "target_arches": "sm_100a,sm_103a",
            "production_arch": "sm_103a",
            "kernel_count": 4,
        }


def _buffer(rows=MOE_FRONT_MAX_M, geometry=PRO_MOE_FRONT_GEOMETRY):
    shared_rows = max(rows, geometry.max_m)
    return SimpleNamespace(
        x=torch.empty(
            (rows, geometry.hidden), dtype=torch.float8_e4m3fn
        ),
        x_sf=torch.empty((rows, geometry.scale_cols), dtype=torch.int32),
        shared_l1_acts_sf=torch.empty_strided(
            (shared_rows, geometry.scale_cols),
            (1, shared_rows),
            dtype=torch.int32,
        ),
        topk_idx=torch.empty((rows, geometry.topk), dtype=torch.int64),
        topk_weights=torch.empty(
            (rows, geometry.topk), dtype=torch.float32
        ),
    )


class _FakeGate:
    def __init__(self, *, is_hash: bool, geometry=PRO_MOE_FRONT_GEOMETRY) -> None:
        self.hash = is_hash
        self.score_func = "sqrtsoftplus"
        self.route_scale = 2.5
        self.weight = torch.empty(
            (geometry.experts, geometry.hidden), dtype=torch.bfloat16
        )
        self.bias = (
            None
            if is_hash
            else torch.empty((geometry.experts,), dtype=torch.float32)
        )
        if is_hash:
            self.tid2eid = torch.zeros((32, geometry.topk), dtype=torch.int64)

    def _weight_bf16(self):
        return self.weight


class _FakeFfn:
    def __init__(self, *, is_hash: bool, geometry=PRO_MOE_FRONT_GEOMETRY) -> None:
        self.dim = geometry.hidden
        self.n_routed_experts = geometry.experts
        self.n_activated_experts = geometry.topk
        self.gate = _FakeGate(is_hash=is_hash, geometry=geometry)
        self.buffer = _buffer(rows=geometry.max_m + 64, geometry=geometry)
        self.geometry = geometry
        self.prepacked_calls = []

    def can_use_native_front(self):
        return True

    def native_front_buffer(self):
        return self.buffer

    def native_front_block_m(self, tokens):
        return 32 if tokens <= 32 else 64 if tokens <= 64 else 128

    def forward_prepacked(self, tokens, device):
        self.prepacked_calls.append((tokens, device))
        return torch.zeros((tokens, self.geometry.hidden), dtype=torch.bfloat16)


class _FakeHc:
    hc_mult = MOE_FRONT_HC_MULT
    norm_eps = 1.0e-6
    hc_eps = 1.0e-6

    def __init__(self) -> None:
        self.post_calls = []

    def post(self, value, residual, post, comb):
        self.post_calls.append((value, residual, post, comb))
        return residual + 1


def _block_and_weights(*, is_hash: bool, geometry=PRO_MOE_FRONT_GEOMETRY):
    ffn = _FakeFfn(is_hash=is_hash, geometry=geometry)
    hc = _FakeHc()
    block = SimpleNamespace(layer_id=3, ffn=ffn, ffn_hc=hc)
    weights = {
        W.v4_hc_ffn_fn: torch.empty(
            (geometry.hc_width, geometry.hc_mult * geometry.hidden),
            dtype=torch.float32,
        ),
        W.v4_hc_ffn_base: torch.empty(
            (geometry.hc_width,), dtype=torch.float32
        ),
        W.v4_hc_ffn_scale: torch.empty((3,), dtype=torch.float32),
        W.v4_ffn_norm: torch.empty((geometry.hidden,), dtype=torch.bfloat16),
    }
    return block, weights


class MegaMoEFrontRuntimeTest(unittest.TestCase):
    def test_accepts_flash_contract_geometry(self) -> None:
        ops = _FakeOps(_geometry(FLASH_MOE_FRONT_GEOMETRY))
        runtime = MegaMoEFrontRuntime(ops_module=ops)
        self.assertIs(
            runtime.require_ops(torch.device("cpu"), FLASH_MOE_FRONT_GEOMETRY),
            ops,
        )

    def test_validates_each_geometry_once_on_a_shared_runtime(self) -> None:
        ops = _FakeOps(_geometry(PRO_MOE_FRONT_GEOMETRY))
        geometries = {
            spec.hidden: _geometry(spec)
            for spec in (PRO_MOE_FRONT_GEOMETRY, FLASH_MOE_FRONT_GEOMETRY)
        }
        calls = []

        def geometry_moe_front(hidden):
            calls.append(hidden)
            return geometries[hidden]

        ops.geometry_moe_front = geometry_moe_front
        runtime = MegaMoEFrontRuntime(ops_module=ops)
        runtime.require_ops(torch.device("cpu"), PRO_MOE_FRONT_GEOMETRY)
        runtime.require_ops(torch.device("cpu"), FLASH_MOE_FRONT_GEOMETRY)
        runtime.require_ops(torch.device("cpu"), FLASH_MOE_FRONT_GEOMETRY)
        self.assertEqual(
            calls,
            [PRO_MOE_FRONT_GEOMETRY.hidden, FLASH_MOE_FRONT_GEOMETRY.hidden],
        )

    def test_rejects_geometry_with_non_fp32_collapse_ssq(self) -> None:
        runtime = MegaMoEFrontRuntime(
            ops_module=_FakeOps(_geometry(collapse_ssq_bits=16))
        )
        with self.assertRaisesRegex(RuntimeError, "geometry mismatch"):
            runtime.require_ops(torch.device("cpu"))

    def test_rejects_stale_build_info(self) -> None:
        ops = _FakeOps()
        ops.build_info_moe_front = lambda: {
            "source_commit": "unknown",
            "source_sha256": "unknown",
            "target_arches": "sm_100a,sm_103a",
            "production_arch": "sm_103a",
            "kernel_count": 4,
        }
        runtime = MegaMoEFrontRuntime(ops_module=ops)
        with self.assertRaisesRegex(RuntimeError, "build info mismatch"):
            runtime.require_ops(torch.device("cpu"))

    def test_rejects_zero_build_identity(self) -> None:
        ops = _FakeOps()
        ops.build_info_moe_front = lambda: {
            "source_commit": "0" * 8,
            "source_sha256": "0" * 64,
            "target_arches": "sm_100a,sm_103a",
            "production_arch": "sm_103a",
            "kernel_count": 4,
        }
        runtime = MegaMoEFrontRuntime(ops_module=ops)
        with self.assertRaisesRegex(RuntimeError, "build info mismatch"):
            runtime.require_ops(torch.device("cpu"))

    def test_rejects_short_zero_copy_buffer(self) -> None:
        runtime = MegaMoEFrontRuntime(ops_module=_FakeOps())
        with self.assertRaisesRegex(TypeError, r"x must be.*\[>=128,7168\]"):
            runtime.validate_buffer(_buffer(rows=64))

    def test_rejects_plan_abi_without_geometry_probe(self) -> None:
        ops = _FakeOps()
        ops.geometry_moe_front = None
        runtime = MegaMoEFrontRuntime(ops_module=ops)
        with self.assertRaisesRegex(RuntimeError, "does not provide"):
            runtime.require_ops(torch.device("cpu"))


class MegaMoEFrontDecodeInputTest(unittest.TestCase):
    def test_stages_decode_into_stable_capacity_without_repeat_output(self) -> None:
        model = V4Transformer.__new__(V4Transformer)
        torch.nn.Module.__init__(model)
        model.hc_mult = MOE_FRONT_HC_MULT
        model._mega_moe_front_runtime = object()
        model._mega_moe_front_hidden_capacity = torch.empty(
            (MOE_FRONT_MAX_M, 1, MOE_FRONT_HC_MULT, MOE_FRONT_HIDDEN),
            dtype=torch.bfloat16,
        )
        model._mega_moe_front_input_ids_capacity = torch.empty(
            (MOE_FRONT_MAX_M,), dtype=torch.int64
        )
        embedded = torch.randn(
            (7, 1, MOE_FRONT_HIDDEN), dtype=torch.bfloat16
        )
        input_ids = torch.arange(7, dtype=torch.int64)

        hidden, staged_ids = model.prepare_moe_front_decode_inputs(
            embedded, input_ids
        )

        self.assertEqual(tuple(hidden.shape), (7, 1, 4, MOE_FRONT_HIDDEN))
        self.assertEqual(
            hidden.data_ptr(), model._mega_moe_front_hidden_capacity.data_ptr()
        )
        self.assertEqual(
            staged_ids.data_ptr(),
            model._mega_moe_front_input_ids_capacity.data_ptr(),
        )
        torch.testing.assert_close(hidden[:, :, 0], embedded)
        torch.testing.assert_close(staged_ids, input_ids)


class MegaMoEFrontAdapterTest(unittest.TestCase):
    def test_rejects_logical_tensor_without_capacity_copy(self) -> None:
        runtime = MegaMoEFrontRuntime(ops_module=_FakeOps())
        block, weights = _block_and_weights(is_hash=False)
        adapter = MegaMoEFrontAdapter(block, weights, runtime)
        hidden = torch.zeros(
            (7, 1, MOE_FRONT_HC_MULT, MOE_FRONT_HIDDEN),
            dtype=torch.bfloat16,
        )
        self.assertFalse(adapter.supports_decode_shape(hidden))

    def _run(self, *, is_hash: bool, geometry=PRO_MOE_FRONT_GEOMETRY):
        ops = _FakeOps(_geometry(geometry))
        runtime = MegaMoEFrontRuntime(ops_module=ops)
        block, weights = _block_and_weights(is_hash=is_hash, geometry=geometry)
        adapter = MegaMoEFrontAdapter(block, weights, runtime)
        hidden_storage = torch.zeros(
            (geometry.max_m, 1, geometry.hc_mult, geometry.hidden),
            dtype=torch.bfloat16,
        )
        hidden = hidden_storage[:7]
        input_ids_storage = torch.arange(geometry.max_m, dtype=torch.int64)
        input_ids = input_ids_storage[:7].view(7, 1)

        first = adapter.forward_ffn_sublayer(block, hidden, input_ids)
        second = adapter.forward_ffn_sublayer(block, hidden, input_ids)

        self.assertEqual(tuple(first.shape), tuple(hidden.shape))
        torch.testing.assert_close(first, hidden + 1)
        torch.testing.assert_close(second, hidden + 1)
        self.assertEqual(len(ops.plan_calls), 1)
        self.assertEqual(
            tuple(ops.plan_calls[0].hidden_states.shape),
            (geometry.max_m, geometry.hc_mult, geometry.hidden),
        )
        self.assertEqual(
            ops.plan_calls[0].hidden_states.data_ptr(), hidden.data_ptr()
        )
        self.assertEqual(block.ffn.prepacked_calls, [(7, hidden.device)] * 2)
        self.assertEqual(len(block.ffn_hc.post_calls), 2)
        return ops, block, input_ids_storage

    def test_flash_front_uses_flash_workspace_and_publication_shapes(self) -> None:
        geometry = FLASH_MOE_FRONT_GEOMETRY
        ops, block, _ = self._run(is_hash=False, geometry=geometry)
        call = ops.learned_calls[0]
        self.assertEqual(
            tuple(call["collapsed"].shape), (geometry.max_m, geometry.hidden)
        )
        self.assertEqual(
            tuple(call["router_logits"].shape),
            (geometry.max_m, geometry.experts),
        )
        self.assertEqual(
            tuple(call["x_sf"].shape), (geometry.max_m, geometry.scale_cols)
        )
        self.assertEqual(call["x_fp8"].data_ptr(), block.ffn.buffer.x.data_ptr())

    def test_learned_front_reuses_context_and_publishes_directly(self) -> None:
        ops, block, _ = self._run(is_hash=False)
        self.assertEqual(len(ops.learned_calls), 2)
        self.assertEqual(len(ops.hash_calls), 0)
        call = ops.learned_calls[0]
        expected_shapes = {
            "x_fp8": (MOE_FRONT_MAX_M, MOE_FRONT_HIDDEN),
            "x_sf": (MOE_FRONT_MAX_M, MOE_FRONT_SCALE_COLS),
            "topk_ids": (MOE_FRONT_MAX_M, MOE_FRONT_TOPK),
            "topk_weights": (MOE_FRONT_MAX_M, MOE_FRONT_TOPK),
        }
        sources = {
            "x_fp8": block.ffn.buffer.x,
            "x_sf": block.ffn.buffer.x_sf,
            "topk_ids": block.ffn.buffer.topk_idx,
            "topk_weights": block.ffn.buffer.topk_weights,
        }
        for name, expected_shape in expected_shapes.items():
            self.assertEqual(tuple(call[name].shape), expected_shape)
            self.assertTrue(call[name].is_contiguous())
            self.assertEqual(call[name].data_ptr(), sources[name].data_ptr())
        self.assertIs(
            call["shared_l1_x_sf"], block.ffn.buffer.shared_l1_acts_sf
        )
        self.assertEqual(call["collapse_ssq"].dtype, torch.float32)

    def test_hash_front_uses_int32_table_and_input_ids(self) -> None:
        ops, block, input_ids_storage = self._run(is_hash=True)
        self.assertEqual(len(ops.learned_calls), 0)
        self.assertEqual(len(ops.hash_calls), 2)
        call = ops.hash_calls[0]
        self.assertEqual(call["tid2eid"].dtype, torch.int32)
        self.assertEqual(tuple(call["input_ids"].shape), (MOE_FRONT_MAX_M,))
        self.assertEqual(call["input_ids"].data_ptr(), input_ids_storage.data_ptr())
        self.assertEqual(
            tuple(call["topk_ids"].shape),
            (MOE_FRONT_MAX_M, MOE_FRONT_TOPK),
        )
        self.assertEqual(
            call["topk_ids"].data_ptr(), block.ffn.buffer.topk_idx.data_ptr()
        )


class MegaMoEFrontBlockRoutingTest(unittest.TestCase):
    def test_decode_uses_complete_native_ffn_sublayer(self) -> None:
        block = Block.__new__(Block)
        torch.nn.Module.__init__(block)
        block.layer_id = 3
        block._mega_csa_adapter = MagicMock()
        block._mega_hca_adapter = None
        block._mega_csa_adapter.supports_decode_shape.return_value = True
        block._mega_csa_adapter.forward_attention_sublayer.side_effect = (
            lambda _block, value, *_args, **_kwargs: value
        )
        block._mega_moe_front_adapter = MagicMock()
        block._mega_moe_front_adapter.supports_decode_shape.return_value = True
        block._mega_moe_front_adapter.forward_ffn_sublayer.side_effect = (
            lambda _block, value, _input_ids: value + 1
        )
        block.ffn_hc = MagicMock()
        block.ffn = MagicMock()
        block.ffn_norm = MagicMock()
        hidden = torch.zeros((2, 1, 4, 8), dtype=torch.bfloat16)

        result = block.forward_decode(
            hidden,
            SimpleNamespace(),
            torch.zeros((2, 1), dtype=torch.int64),
        )

        torch.testing.assert_close(result, hidden + 1)
        block._mega_moe_front_adapter.forward_ffn_sublayer.assert_called_once()
        block.ffn_hc.pre.assert_not_called()
        block.ffn.assert_not_called()
        block.ffn_norm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
