#!/usr/bin/env python3
"""Export a SEA-Nav Isaac Sim/Isaac Lab `.pt` policy checkpoint to ONNX.

Typical usage:

    python scripts/export_onnx.py \
        --input logs/sea_nav/Go2_pos_rough/2026-07-08_15-39-29/model_1000.pt \
        --output logs/sea_nav/Go2_pos_rough/2026-07-08_15-39-29/policy.onnx \
        --verify

The SEA-Nav training scripts save RSL-RL checkpoints as dictionaries containing
`model_state_dict`. This exporter rebuilds the local actor-critic module from
that state dict and exports only deterministic inference actions.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEA_NAV_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RSL_RL_ROOT = SEA_NAV_ROOT / "training" / "rsl_rl"
sys.path.insert(0, str(LOCAL_RSL_RL_ROOT))

from rsl_rl.modules.actor_critic import ActorCritic  # noqa: E402
from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic  # noqa: E402


POLICY_CLASSES: dict[str, type[nn.Module]] = {
    "ActorCritic": ActorCritic,
    "DifferentiableSafeActorCritic": DifferentiableSafeActorCritic,
}


class InferencePolicy(nn.Module):
    """Small ONNX-facing wrapper that calls the deterministic policy path."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if isinstance(self.policy, DifferentiableSafeActorCritic):
            return self._forward_safe_actor_critic(observations)
        if hasattr(self.policy, "act_inference"):
            return self.policy.act_inference(observations)
        return self.policy(observations)

    def _forward_safe_actor_critic(self, observations: torch.Tensor) -> torch.Tensor:
        """ONNX-friendly equivalent of DifferentiableSafeActorCritic.act_inference."""

        obs_buf = observations[:, -self.policy.num_obs_one_step :]
        rays = obs_buf[:, self.policy.num_props : self.policy.num_props + self.policy.num_rays]

        latent = self.policy.encoder(observations)
        obs_cat = torch.cat((obs_buf, latent.detach()), dim=-1)
        shared_features = self.policy.backbone(obs_cat)
        u_bar = self.policy.nav_head(shared_features)
        alpha = F.softplus(self.policy.alpha_head(shared_features))

        rays_real = torch.exp(rays * 0.6931471805599453)
        return self.policy.cbf_layer(u_bar, rays_real, alpha)


def _torch_load(path: Path, device: torch.device) -> Any:
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(checkpoint, nn.Module):
        return None
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    for key in ("model_state_dict", "state_dict", "policy_state_dict", "actor_critic_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint

    raise KeyError(
        "Could not find model weights in checkpoint. Expected one of "
        "`model_state_dict`, `state_dict`, `policy_state_dict`, or a raw state dict."
    )


def _infer_policy_class(state_dict: dict[str, torch.Tensor], requested: str) -> type[nn.Module]:
    if requested != "auto":
        return POLICY_CLASSES[requested]
    if "cbf_layer.ray_unit_vectors" in state_dict or any(key.startswith("alpha_head.") for key in state_dict):
        return DifferentiableSafeActorCritic
    if any(key.startswith("actor.") for key in state_dict):
        return ActorCritic
    raise ValueError("Could not infer policy class. Pass --policy-class explicitly.")


def _linear_layers(state_dict: dict[str, torch.Tensor], prefix: str) -> list[tuple[int, torch.Tensor]]:
    layers: list[tuple[int, torch.Tensor]] = []
    marker = f"{prefix}."
    for key, tensor in state_dict.items():
        if not key.startswith(marker) or not key.endswith(".weight") or tensor.ndim != 2:
            continue
        layer_name = key[len(marker) : -len(".weight")]
        if layer_name.isdigit():
            layers.append((int(layer_name), tensor))
    return sorted(layers, key=lambda item: item[0])


def _infer_model_kwargs(
    state_dict: dict[str, torch.Tensor],
    policy_class: type[nn.Module],
    args: argparse.Namespace,
) -> dict[str, Any]:
    is_safe_policy = policy_class is DifferentiableSafeActorCritic
    actor_prefix = "backbone" if is_safe_policy else "actor"
    actor_layers = _linear_layers(state_dict, actor_prefix)
    encoder_layers = _linear_layers(state_dict, "encoder")
    critic_layers = _linear_layers(state_dict, "critic")
    if not actor_layers or not encoder_layers:
        raise ValueError("Checkpoint does not look like a SEA-Nav actor-critic state dict.")

    latent_dim = int(encoder_layers[-1][1].shape[0])
    one_step_obs = int(actor_layers[0][1].shape[1]) - latent_dim
    if one_step_obs <= 2:
        raise ValueError(f"Invalid inferred one-step observation width: {one_step_obs}")

    if args.num_rays is not None:
        num_rays = args.num_rays
    elif "cbf_layer.ray_unit_vectors" in state_dict:
        num_rays = int(state_dict["cbf_layer.ray_unit_vectors"].shape[0])
    else:
        raise ValueError("Could not infer ray count. Pass --num-rays.")

    num_props = args.num_props if args.num_props is not None else one_step_obs - num_rays - 2
    if num_props <= 0:
        raise ValueError(
            f"Invalid inferred num_props={num_props}. Check --num-rays/--num-props for this checkpoint."
        )

    encoder_input_dim = int(encoder_layers[0][1].shape[1])
    his_len = args.his_len if args.his_len is not None else encoder_input_dim // one_step_obs
    if encoder_input_dim != one_step_obs * his_len:
        raise ValueError(
            "Observation history dimensions do not match: "
            f"encoder input {encoder_input_dim}, one-step {one_step_obs}, his_len {his_len}."
        )

    if args.num_actions is not None:
        num_actions = args.num_actions
    elif "std" in state_dict:
        num_actions = int(state_dict["std"].shape[0])
    elif is_safe_policy and "nav_head.2.weight" in state_dict:
        num_actions = int(state_dict["nav_head.2.weight"].shape[0])
    else:
        num_actions = int(actor_layers[-1][1].shape[0])

    encoder_hidden_dims = [int(weight.shape[0]) for _, weight in encoder_layers[:-1]]
    actor_hidden_dims = [int(weight.shape[0]) for _, weight in actor_layers]
    if not is_safe_policy and actor_hidden_dims and actor_hidden_dims[-1] == num_actions:
        actor_hidden_dims.pop()
    critic_hidden_dims = [int(weight.shape[0]) for _, weight in critic_layers]
    if critic_hidden_dims and critic_hidden_dims[-1] == 1:
        critic_hidden_dims.pop()

    return {
        "num_actions": num_actions,
        "actor_hidden_dims": actor_hidden_dims,
        "critic_hidden_dims": critic_hidden_dims,
        "encoder_hidden_dims": encoder_hidden_dims,
        "activation": args.activation,
        "init_noise_std": 1.5,
        "num_props": num_props,
        "num_rays": num_rays,
        "his_len": his_len,
    }


def load_policy(checkpoint_path: Path, args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, tuple[int, int]]:
    checkpoint = _torch_load(checkpoint_path, device)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint.to(device).eval()
        if args.input_shape is None:
            raise ValueError("--input-shape is required when exporting a serialized nn.Module.")
        return model, _parse_shape(args.input_shape)

    state_dict = _state_dict_from_checkpoint(checkpoint)
    policy_class = _infer_policy_class(state_dict, args.policy_class)
    model_kwargs = _infer_model_kwargs(state_dict, policy_class, args)
    model = policy_class(**model_kwargs).to(device).eval()
    model.load_state_dict(state_dict, strict=True)

    inferred_shape = (1, (model_kwargs["num_rays"] + model_kwargs["num_props"] + 2) * model_kwargs["his_len"])
    input_shape = _parse_shape(args.input_shape) if args.input_shape is not None else inferred_shape
    return model, input_shape


def _parse_shape(shape: str) -> tuple[int, int]:
    values = tuple(int(part.strip()) for part in shape.split(",") if part.strip())
    if len(values) != 2:
        raise ValueError("--input-shape must have two dimensions, for example `1,550`.")
    if any(value <= 0 for value in values):
        raise ValueError("--input-shape dimensions must be positive.")
    return values


def export_to_onnx(
    policy: nn.Module,
    output_path: Path,
    input_shape: tuple[int, int],
    device: torch.device,
    dynamic_batch: bool,
    opset_version: int,
) -> tuple[InferencePolicy, torch.Tensor]:
    wrapper = InferencePolicy(policy).to(device).eval()
    dummy_input = torch.zeros(*input_shape, dtype=torch.float32, device=device)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"observations": {0: "batch_size"}, "actions": {0: "batch_size"}}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            input_names=["observations"],
            output_names=["actions"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )
    return wrapper, dummy_input


def verify_onnx(output_path: Path, wrapper: nn.Module, dummy_input: torch.Tensor, tolerance: float) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnx and/or onnxruntime is not installed; skipping verification.")
        return

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    with torch.inference_mode():
        torch_output = wrapper(dummy_input).detach().cpu().numpy()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.detach().cpu().numpy()}
    onnx_output = session.run(None, ort_inputs)[0]
    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    if max_abs_diff > tolerance:
        raise RuntimeError(
            f"ONNX verification failed: max abs diff {max_abs_diff:.3e} exceeds tolerance {tolerance:.3e}."
        )
    print(f"[INFO] ONNX verification passed. Max abs diff: {max_abs_diff:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input `.pt` checkpoint/model path.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output `.onnx` path.")
    parser.add_argument(
        "--input-shape",
        type=str,
        default=None,
        help="Optional input tensor shape, for example `1,550`. Inferred for SEA-Nav checkpoints.",
    )
    parser.add_argument(
        "--policy-class",
        choices=("auto", *POLICY_CLASSES.keys()),
        default="auto",
        help="Policy class to rebuild from a state dict. Default: auto.",
    )
    parser.add_argument("--num-rays", type=int, default=None, help="Override inferred LiDAR/ray observation count.")
    parser.add_argument("--num-props", type=int, default=None, help="Override inferred proprioceptive observation count.")
    parser.add_argument("--his-len", type=int, default=None, help="Override inferred observation history length.")
    parser.add_argument("--num-actions", type=int, default=None, help="Override inferred action dimension.")
    parser.add_argument("--activation", type=str, default="elu", help="Activation used by the trained policy.")
    parser.add_argument("--device", type=str, default="cpu", help="Device used for export. Default: cpu.")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version. Default: 13.")
    parser.add_argument("--dynamic-batch", action="store_true", help="Export a dynamic batch dimension.")
    parser.add_argument("--verify", action="store_true", help="Check ONNX and compare ONNX Runtime vs PyTorch output.")
    parser.add_argument("--verify-tolerance", type=float, default=1e-5, help="Max allowed verification difference.")
    args = parser.parse_args()

    checkpoint_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Input checkpoint does not exist: {checkpoint_path}")

    device = torch.device(args.device)
    policy, input_shape = load_policy(checkpoint_path, args, device)
    wrapper, dummy_input = export_to_onnx(
        policy=policy,
        output_path=output_path,
        input_shape=input_shape,
        device=device,
        dynamic_batch=args.dynamic_batch,
        opset_version=args.opset,
    )

    print(f"[INFO] Exported ONNX policy: {output_path}")
    print(f"[INFO] Input shape: {input_shape}")
    if args.verify:
        verify_onnx(output_path, wrapper, dummy_input, args.verify_tolerance)


if __name__ == "__main__":
    main()
