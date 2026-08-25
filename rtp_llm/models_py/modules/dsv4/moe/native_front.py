"""Production adapter for the DeepSeek-V4 Pro four-kernel MoE front.

The CUDA implementation is supplied by a matching ``rtp-kernel.dsv4_mega``
build. This module owns model/runtime policy only: ABI validation, stable
workspaces, pointer-bound plans, supported-shape routing, and the handoff into
MegaMoE-SE's existing symmetric buffer.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range


MOE_FRONT_ABI_VERSION = 1
MOE_FRONT_KERNEL_CONTRACT_VERSION = 1
MOE_FRONT_HIDDEN = 7168
MOE_FRONT_HC_MULT = 4
MOE_FRONT_HC_WIDTH = 24
MOE_FRONT_EXPERTS = 384
MOE_FRONT_TOPK = 6
MOE_FRONT_MAX_M = 128
MOE_FRONT_SCALE_COLS = MOE_FRONT_HIDDEN // 128


@dataclass
class MegaMoEFrontWorkspace:
    collapsed: torch.Tensor
    collapse_ssq: torch.Tensor
    normalized_mix: torch.Tensor
    normalized: torch.Tensor
    router_logits: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor


@dataclass
class _FrontPlanEntry:
    plan: Any
    hidden: torch.Tensor
    hc_fn: torch.Tensor


class MegaMoEFrontRuntime:
    """Model-level graph-stable workspace and pointer-bound context cache."""

    def __init__(self, *, ops_module: Any | None = None) -> None:
        self._ops = ops_module
        self._injected_ops = ops_module is not None
        self._runtime_checked = False
        self._workspaces: Dict[Tuple[str, int, int], MegaMoEFrontWorkspace] = {}
        self._plans: Dict[int, Dict[tuple, _FrontPlanEntry]] = {}
        self._validated_buffers: set[int] = set()

    @property
    def allows_cpu_for_test(self) -> bool:
        return self._injected_ops

    @staticmethod
    def _capturing(device: torch.device) -> bool:
        return device.type == "cuda" and torch.cuda.is_current_stream_capturing()

    def require_ops(self, device: torch.device) -> Any:
        if self._runtime_checked:
            assert self._ops is not None
            return self._ops
        if not self._injected_ops:
            capability = torch.cuda.get_device_capability(device)
            if capability not in ((10, 0), (10, 3)):
                raise RuntimeError(
                    "DSV4 native MoE front requires sm_100a or sm_103a, "
                    f"got sm_{capability[0]}{capability[1]}"
                )
            from rtp_kernel import dsv4_mega

            self._ops = dsv4_mega
        assert self._ops is not None
        required = ("Dsv4MoeFrontPlan", "geometry_moe_front")
        missing = [
            name
            for name in required
            if not callable(getattr(self._ops, name, None))
        ]
        if missing:
            raise RuntimeError(
                "rtp-kernel does not provide DSV4 MoE front ABI v1: "
                + ", ".join(missing)
            )
        required_parameters = {
            "Dsv4MoeFrontPlan": ("hidden_states", "hc_fn", "logical_m"),
            "Dsv4MoeFrontPlan.run_learned_out": (
                "hc_base",
                "hc_scale",
                "ffn_norm_weight",
                "router_weight",
                "correction_bias",
                "collapsed",
                "collapse_ssq",
                "normalized_mix",
                "normalized",
                "x_fp8",
                "x_sf",
                "shared_l1_x_sf",
                "topk_ids",
                "topk_weights",
                "post",
                "comb",
                "shared_block_m",
                "router_logits",
                "norm_eps",
                "hc_eps",
                "route_scale",
                "use_pdl",
            ),
            "Dsv4MoeFrontPlan.run_hash_out": (
                "hc_base",
                "hc_scale",
                "ffn_norm_weight",
                "router_weight",
                "input_ids",
                "tid2eid",
                "collapsed",
                "collapse_ssq",
                "normalized_mix",
                "normalized",
                "x_fp8",
                "x_sf",
                "shared_l1_x_sf",
                "router_logits",
                "topk_ids",
                "topk_weights",
                "post",
                "comb",
                "shared_block_m",
                "norm_eps",
                "hc_eps",
                "route_scale",
                "use_pdl",
            ),
        }
        incompatible = []
        for function_name, parameters in required_parameters.items():
            target: Any = self._ops
            for component in function_name.split("."):
                target = getattr(target, component)
            try:
                signature = inspect.signature(target)
            except (TypeError, ValueError) as exc:
                incompatible.append(f"{function_name} is not introspectable: {exc}")
                continue
            absent = [name for name in parameters if name not in signature.parameters]
            if absent:
                incompatible.append(f"{function_name} missing {','.join(absent)}")
        if incompatible:
            raise RuntimeError(
                "rtp-kernel DSV4 MoE front ABI is incompatible: "
                + "; ".join(incompatible)
            )
        geometry = self._ops.geometry_moe_front()
        expected = {
            "abi_version": MOE_FRONT_ABI_VERSION,
            "kernel_contract_version": MOE_FRONT_KERNEL_CONTRACT_VERSION,
            "hidden": MOE_FRONT_HIDDEN,
            "hc_mult": MOE_FRONT_HC_MULT,
            "hc_width": MOE_FRONT_HC_WIDTH,
            "experts": MOE_FRONT_EXPERTS,
            "topk": MOE_FRONT_TOPK,
            "max_m": MOE_FRONT_MAX_M,
            "scale_cols": MOE_FRONT_SCALE_COLS,
            "collapse_ssq_bits": 32,
        }
        mismatched = {
            key: geometry.get(key)
            for key, value in expected.items()
            if geometry.get(key) != value
        }
        if mismatched:
            raise RuntimeError(
                "rtp-kernel DSV4 MoE front geometry mismatch: "
                f"got {mismatched}, expected {expected}"
            )
        self._runtime_checked = True
        return self._ops

    def validate_buffer(self, buffer: Any) -> None:
        """Validate the zero-copy DeepGEMM publication ABI once per buffer."""
        key = id(buffer)
        if key in self._validated_buffers:
            return

        expected = (
            (
                "x",
                torch.float8_e4m3fn,
                MOE_FRONT_HIDDEN,
                (MOE_FRONT_HIDDEN, 1),
            ),
            ("x_sf", torch.int32, MOE_FRONT_SCALE_COLS, (MOE_FRONT_SCALE_COLS, 1)),
            ("topk_idx", torch.int64, MOE_FRONT_TOPK, (MOE_FRONT_TOPK, 1)),
            (
                "topk_weights",
                torch.float32,
                MOE_FRONT_TOPK,
                (MOE_FRONT_TOPK, 1),
            ),
        )
        for name, dtype, columns, trailing_stride in expected:
            tensor = getattr(buffer, name, None)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"MegaMoE-SE buffer is missing tensor {name!r}")
            valid = (
                tensor.dtype == dtype
                and tensor.dim() == 2
                and int(tensor.shape[0]) >= MOE_FRONT_MAX_M
                and int(tensor.shape[1]) == columns
                and tuple(tensor.stride()) == trailing_stride
            )
            if not valid:
                raise TypeError(
                    f"MegaMoE-SE buffer {name} must be {dtype} row-major "
                    f"[>={MOE_FRONT_MAX_M},{columns}], got dtype={tensor.dtype}, "
                    f"shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}"
                )

        shared = getattr(buffer, "shared_l1_acts_sf", None)
        if not isinstance(shared, torch.Tensor):
            raise TypeError("MegaMoE-SE buffer is missing tensor 'shared_l1_acts_sf'")
        valid_shared = (
            shared.dtype == torch.int32
            and shared.dim() == 2
            and int(shared.shape[0]) >= MOE_FRONT_MAX_M
            and int(shared.shape[1]) == MOE_FRONT_SCALE_COLS
            and int(shared.stride(0)) == 1
            and int(shared.stride(1)) == int(shared.shape[0])
        )
        if not valid_shared:
            raise TypeError(
                "MegaMoE-SE buffer shared_l1_acts_sf must be INT32 column-major "
                f"[>={MOE_FRONT_MAX_M},{MOE_FRONT_SCALE_COLS}], got "
                f"dtype={shared.dtype}, shape={tuple(shared.shape)}, "
                f"stride={tuple(shared.stride())}"
            )
        self._validated_buffers.add(key)

    def workspace(self, device: torch.device, buffer: Any) -> MegaMoEFrontWorkspace:
        device_index = -1 if device.index is None else int(device.index)
        key = (device.type, device_index, id(buffer))
        workspace = self._workspaces.get(key)
        if workspace is not None:
            return workspace
        if self._capturing(device):
            raise RuntimeError(
                "DSV4 native MoE front workspace must be allocated before "
                "CUDA graph capture"
            )
        workspace = MegaMoEFrontWorkspace(
            collapsed=torch.empty(
                (MOE_FRONT_MAX_M, MOE_FRONT_HIDDEN),
                dtype=torch.bfloat16,
                device=device,
            ),
            collapse_ssq=torch.empty(
                (MOE_FRONT_MAX_M,), dtype=torch.float32, device=device
            ),
            normalized_mix=torch.empty(
                (MOE_FRONT_MAX_M, MOE_FRONT_HC_WIDTH),
                dtype=torch.float32,
                device=device,
            ),
            normalized=torch.empty(
                (MOE_FRONT_MAX_M, MOE_FRONT_HIDDEN),
                dtype=torch.bfloat16,
                device=device,
            ),
            router_logits=torch.empty(
                (MOE_FRONT_MAX_M, MOE_FRONT_EXPERTS),
                dtype=torch.float32,
                device=device,
            ),
            post=torch.empty(
                (MOE_FRONT_MAX_M, MOE_FRONT_HC_MULT),
                dtype=torch.float32,
                device=device,
            ),
            comb=torch.empty(
                (MOE_FRONT_MAX_M, MOE_FRONT_HC_MULT, MOE_FRONT_HC_MULT),
                dtype=torch.float32,
                device=device,
            ),
        )
        self._workspaces[key] = workspace
        return workspace

    def plan(
        self,
        ops: Any,
        layer_id: int,
        hidden: torch.Tensor,
        hc_fn: torch.Tensor,
        logical_m: int,
    ) -> Any:
        key = (
            hidden.device.type,
            -1 if hidden.device.index is None else int(hidden.device.index),
            int(logical_m),
            int(hidden.data_ptr()),
            int(hc_fn.data_ptr()),
        )
        layer_plans = self._plans.setdefault(layer_id, {})
        entry = layer_plans.get(key)
        if entry is not None:
            return entry.plan
        if self._capturing(hidden.device):
            raise RuntimeError(
                "DSV4 native MoE front plan must be created before CUDA graph capture"
            )
        plan = ops.Dsv4MoeFrontPlan(hidden, hc_fn, int(logical_m))
        layer_plans[key] = _FrontPlanEntry(plan, hidden, hc_fn)
        return plan


class MegaMoEFrontAdapter:
    """Replace the complete Pro decode FFN front and preserve mHC post."""

    def __init__(self, block, layer_weights: Dict, runtime: MegaMoEFrontRuntime):
        from rtp_llm.utils.model_weight import W

        if not block.ffn.can_use_native_front():
            strategy = getattr(getattr(block.ffn, "_strategy", None), "name", "unknown")
            raise RuntimeError(
                "DSV4 native MoE front requires MegaMoEStrategySE; "
                f"selected strategy={strategy!r}"
            )
        gate = block.ffn.gate
        expected = (
            ("hidden", int(block.ffn.dim), MOE_FRONT_HIDDEN),
            ("experts", int(block.ffn.n_routed_experts), MOE_FRONT_EXPERTS),
            ("topk", int(block.ffn.n_activated_experts), MOE_FRONT_TOPK),
            ("hc_mult", int(block.ffn_hc.hc_mult), MOE_FRONT_HC_MULT),
        )
        problems = [
            f"{name}={actual} (expected {wanted})"
            for name, actual, wanted in expected
            if actual != wanted
        ]
        if problems:
            raise ValueError(
                "DSV4 native MoE front geometry mismatch: " + "; ".join(problems)
            )
        if gate.score_func != "sqrtsoftplus":
            raise ValueError(
                "DSV4 native MoE front requires score_func='sqrtsoftplus', "
                f"got {gate.score_func!r}"
            )

        self.layer_id = int(block.layer_id)
        self.runtime = runtime
        self.hc_fn = layer_weights[W.v4_hc_ffn_fn].contiguous()
        self.hc_base = layer_weights[W.v4_hc_ffn_base].contiguous()
        self.hc_scale = layer_weights[W.v4_hc_ffn_scale].contiguous()
        self.ffn_norm_weight = layer_weights[W.v4_ffn_norm].contiguous()
        self.router_weight = gate._weight_bf16().contiguous()
        self.correction_bias = gate.bias
        self.tid2eid = getattr(gate, "tid2eid", None)
        if self.tid2eid is not None:
            self.tid2eid = self.tid2eid.to(torch.int32).contiguous()
        self.is_hash = bool(gate.hash)
        self.route_scale = float(gate.route_scale)
        self.norm_eps = float(block.ffn_hc.norm_eps)
        self.hc_eps = float(block.ffn_hc.hc_eps)
        self._validate_weights()

    def _validate_weights(self) -> None:
        tensors = (
            (
                "hc_fn",
                self.hc_fn,
                (MOE_FRONT_HC_WIDTH, MOE_FRONT_HC_MULT * MOE_FRONT_HIDDEN),
                torch.float32,
            ),
            ("hc_base", self.hc_base, (MOE_FRONT_HC_WIDTH,), torch.float32),
            ("hc_scale", self.hc_scale, (3,), torch.float32),
            (
                "ffn_norm_weight",
                self.ffn_norm_weight,
                (MOE_FRONT_HIDDEN,),
                torch.bfloat16,
            ),
            (
                "router_weight",
                self.router_weight,
                (MOE_FRONT_EXPERTS, MOE_FRONT_HIDDEN),
                torch.bfloat16,
            ),
        )
        for name, tensor, shape, dtype in tensors:
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise TypeError(
                    f"DSV4 native MoE front {name} must be contiguous {dtype} "
                    f"{shape}, got dtype={tensor.dtype}, shape={tuple(tensor.shape)}, "
                    f"stride={tuple(tensor.stride())}"
                )
        if self.is_hash:
            if self.correction_bias is not None or self.tid2eid is None:
                raise ValueError("HashMoE native front requires tid2eid and no bias")
            if (
                self.tid2eid.dtype != torch.int32
                or self.tid2eid.shape[1] != MOE_FRONT_TOPK
            ):
                raise TypeError("HashMoE tid2eid must be contiguous INT32 [vocab,6]")
        else:
            if self.correction_bias is None or self.tid2eid is not None:
                raise ValueError("learned native front requires correction bias only")
            if (
                self.correction_bias.dtype != torch.float32
                or tuple(self.correction_bias.shape) != (MOE_FRONT_EXPERTS,)
                or not self.correction_bias.is_contiguous()
            ):
                raise TypeError("learned Router bias must be contiguous FP32 [384]")

    @staticmethod
    def _has_row_capacity(tensor: torch.Tensor, row_elements: int) -> bool:
        if not tensor.is_contiguous():
            return False
        available_bytes = tensor.untyped_storage().nbytes() - (
            int(tensor.storage_offset()) * tensor.element_size()
        )
        required_bytes = MOE_FRONT_MAX_M * row_elements * tensor.element_size()
        return available_bytes >= required_bytes

    @staticmethod
    def _capacity_view(tensor: torch.Tensor, *row_shape: int) -> torch.Tensor:
        row_elements = 1
        for extent in row_shape:
            row_elements *= extent
        if not MegaMoEFrontAdapter._has_row_capacity(tensor, row_elements):
            raise ValueError(
                "DSV4 native MoE front requires graph-stable storage capacity "
                f"for {MOE_FRONT_MAX_M} rows of shape {row_shape}"
            )
        trailing_strides = []
        stride = 1
        for extent in reversed(row_shape):
            trailing_strides.append(stride)
            stride *= extent
        return tensor.as_strided(
            (MOE_FRONT_MAX_M, *row_shape),
            (row_elements, *reversed(trailing_strides)),
        )

    def supports_decode_shape(
        self, hidden: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> bool:
        shape_supported = (
            hidden.dim() == 4
            and int(hidden.shape[1]) == 1
            and tuple(hidden.shape[2:]) == (MOE_FRONT_HC_MULT, MOE_FRONT_HIDDEN)
            and 1 <= int(hidden.shape[0]) <= MOE_FRONT_MAX_M
            and hidden.dtype == torch.bfloat16
            and hidden.is_contiguous()
        )
        if not shape_supported or not self._has_row_capacity(
            hidden, MOE_FRONT_HC_MULT * MOE_FRONT_HIDDEN
        ):
            return False
        if not self.is_hash:
            return True
        return bool(
            isinstance(input_ids, torch.Tensor)
            and input_ids.dtype == torch.int64
            and input_ids.numel() == int(hidden.shape[0])
            and self._has_row_capacity(input_ids, 1)
        )

    def forward_ffn_sublayer(
        self, block, hidden: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        if not self.supports_decode_shape(hidden, input_ids):
            raise ValueError(
                "DSV4 native MoE front requires [M,1,4,7168] with stable "
                "128-row storage capacity (and int64 input IDs with the same "
                f"capacity for HashMoE); got hidden={tuple(hidden.shape)}"
            )
        if hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
            raise TypeError("DSV4 native MoE front hidden must be contiguous BF16")
        if not hidden.is_cuda and not self.runtime.allows_cpu_for_test:
            raise TypeError("DSV4 native MoE front hidden must be CUDA")
        m = int(hidden.shape[0])
        flat_hidden = hidden.view(m, MOE_FRONT_HC_MULT, MOE_FRONT_HIDDEN)
        hidden_capacity = self._capacity_view(
            flat_hidden, MOE_FRONT_HC_MULT, MOE_FRONT_HIDDEN
        )
        ops = self.runtime.require_ops(hidden.device)
        buffer = block.ffn.native_front_buffer()
        self.runtime.validate_buffer(buffer)
        workspace = self.runtime.workspace(hidden.device, buffer)
        plan = self.runtime.plan(
            ops, self.layer_id, hidden_capacity, self.hc_fn, m
        )
        shared_block_m = block.ffn.native_front_block_m(m)
        x_fp8_capacity = self._capacity_view(buffer.x, MOE_FRONT_HIDDEN)
        x_sf_capacity = self._capacity_view(buffer.x_sf, MOE_FRONT_SCALE_COLS)
        topk_ids_capacity = self._capacity_view(buffer.topk_idx, MOE_FRONT_TOPK)
        topk_weights_capacity = self._capacity_view(
            buffer.topk_weights, MOE_FRONT_TOPK
        )
        common = dict(
            hc_base=self.hc_base,
            hc_scale=self.hc_scale,
            ffn_norm_weight=self.ffn_norm_weight,
            router_weight=self.router_weight,
            collapsed=workspace.collapsed,
            collapse_ssq=workspace.collapse_ssq,
            normalized_mix=workspace.normalized_mix,
            normalized=workspace.normalized,
            x_fp8=x_fp8_capacity,
            x_sf=x_sf_capacity,
            shared_l1_x_sf=buffer.shared_l1_acts_sf,
            router_logits=workspace.router_logits,
            topk_ids=topk_ids_capacity,
            topk_weights=topk_weights_capacity,
            post=workspace.post,
            comb=workspace.comb,
            shared_block_m=shared_block_m,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
            route_scale=self.route_scale,
            use_pdl=True,
        )
        with record_function_range("dsv4.moe.native_four_kernel_front"):
            if self.is_hash:
                input_ids_capacity = self._capacity_view(
                    input_ids.view(-1), 1
                ).view(-1)
                plan.run_hash_out(
                    input_ids=input_ids_capacity,
                    tid2eid=self.tid2eid,
                    **common,
                )
            else:
                plan.run_learned_out(
                    correction_bias=self.correction_bias,
                    **common,
                )
        ffn_out = block.ffn.forward_prepacked(m, hidden.device)
        post = workspace.post[:m].view(m, 1, MOE_FRONT_HC_MULT, 1)
        comb = workspace.comb[:m].view(
            m, 1, MOE_FRONT_HC_MULT, MOE_FRONT_HC_MULT
        )
        return block.ffn_hc.post(
            ffn_out.view(m, 1, MOE_FRONT_HIDDEN), hidden, post, comb
        )


__all__ = ["MegaMoEFrontAdapter", "MegaMoEFrontRuntime"]
