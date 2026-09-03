"""Compatibility patch for OpenRLHF 0.9.3 LoRA weight sync to vLLM.

OpenRLHF's CUDA-IPC sync sends ``PeftModel.named_parameters()`` directly to
vLLM.  PEFT parameter names have a ``base_model.model.`` prefix and include
adapter tensors, while vLLM expects the original model names and dense,
adapter-merged weights.  This module supplies a Ray worker setup hook that
installs a narrowly scoped replacement for the sync method.
"""

from __future__ import annotations


_PEFT_PREFIX = "base_model.model."
_ADAPTER_MARKERS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_embedding_A.",
    ".lora_embedding_B.",
    ".lora_magnitude_vector.",
)


def vllm_weight_name(peft_name: str) -> str | None:
    """Return the dense vLLM weight name, or ``None`` for adapter tensors."""

    if any(marker in peft_name for marker in _ADAPTER_MARKERS):
        return None
    if peft_name.startswith(_PEFT_PREFIX):
        peft_name = peft_name[len(_PEFT_PREFIX) :]
    # PEFT wraps targeted Linear modules in a tuner layer and moves the
    # original dense parameter under ``base_layer``.  vLLM retains the
    # unwrapped module layout.
    return peft_name.replace(".base_layer.", ".")


def merged_vllm_parameters(model):
    """List base parameters with names accepted by the vLLM model loader."""

    parameters = []
    seen = set()
    for peft_name, parameter in model.named_parameters():
        name = vllm_weight_name(peft_name)
        if name is None:
            continue
        if name in seen:
            raise ValueError(f"duplicate vLLM weight name after PEFT mapping: {name}")
        seen.add(name)
        parameters.append((name, parameter))
    return parameters


def _adapter_owner(model):
    if hasattr(model, "merge_adapter") and hasattr(model, "unmerge_adapter"):
        return model
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "merge_adapter") and hasattr(
        base_model, "unmerge_adapter"
    ):
        return base_model
    raise TypeError("LoRA sync requested, but the actor is not a mergeable PEFT model")


def _broadcast_merged_lora_to_vllm(self):
    """OpenRLHF broadcast replacement used only when LoRA is enabled."""

    import deepspeed
    import ray
    import torch

    from openrlhf.utils.distributed_util import torch_dist_barrier_and_cuda_sync
    from openrlhf.trainer.ray.utils import get_physical_gpu_id

    use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
    cache_reset_refs = []
    if use_prefix_cache and torch.distributed.get_rank() == 0:
        cache_reset_refs = [
            engine.reset_prefix_cache.remote() for engine in self.vllm_engines
        ]

    torch.cuda.empty_cache()
    model = self.actor.model.module
    adapter = _adapter_owner(model)
    adapter.merge_adapter()
    try:
        named_parameters = merged_vllm_parameters(model)
        if not named_parameters:
            raise ValueError("PEFT actor exposed no dense parameters for vLLM sync")
        num_params = len(named_parameters)

        def broadcast_param(name, parameter, count):
            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            if torch.distributed.get_rank() == 0:
                shape = (
                    parameter.shape
                    if self.strategy.args.zero_stage != 3
                    else parameter.ds_shape
                )
                refs = [
                    engine.update_weight.remote(
                        name,
                        dtype=parameter.dtype,
                        shape=shape,
                        empty_cache=count == num_params,
                    )
                    for engine in self.vllm_engines
                ]
                if use_ray:
                    import ray.util.collective as collective

                    collective.broadcast(
                        parameter.data, 0, group_name=self._model_update_group
                    )
                else:
                    self._model_update_group.broadcast(
                        parameter.data, src=0, stream=torch.cuda.current_stream()
                    )
                ray.get(refs)

        def handle_cuda_ipc(name, parameter, count):
            from torch.multiprocessing.reductions import reduce_tensor

            weight = parameter.data.clone()
            ipc_handle = {get_physical_gpu_id(): reduce_tensor(weight)}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)
            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for value in ipc_handle_list:
                    ipc_handles.update(value)
                shape = (
                    parameter.shape
                    if self.strategy.args.zero_stage != 3
                    else parameter.ds_shape
                )
                refs = [
                    engine.update_weight_cuda_ipc.remote(
                        name,
                        dtype=parameter.dtype,
                        shape=shape,
                        ipc_handles=ipc_handles,
                        empty_cache=count == num_params,
                    )
                    for engine in self.vllm_engines
                ]
                ray.get(refs)
            torch_dist_barrier_and_cuda_sync()

        for count, (name, parameter) in enumerate(named_parameters, start=1):
            if self.strategy.args.ds_tensor_parallel_size > 1:
                gathered = deepspeed.module_inject.layers.GatherReplacedLayerParams(
                    [parameter], model, enabled=True
                )
            else:
                gathered = deepspeed.zero.GatheredParameters(
                    [parameter], enabled=self.strategy.args.zero_stage == 3
                )
            with gathered:
                if self.use_cuda_ipc:
                    handle_cuda_ipc(name, parameter, count)
                else:
                    broadcast_param(name, parameter, count)
    finally:
        adapter.unmerge_adapter()

    if cache_reset_refs:
        ray.get(cache_reset_refs)
    torch.cuda.empty_cache()
    torch_dist_barrier_and_cuda_sync()


def install_openrlhf_lora_vllm_sync_patch() -> None:
    """Install the patch in the driver or a Ray worker process."""

    from openrlhf.trainer.ray.ppo_actor import ActorPPOTrainer

    if getattr(ActorPPOTrainer, "_nashrs_lora_sync_patch", False):
        return
    ActorPPOTrainer.broadcast_to_vllm = _broadcast_merged_lora_to_vllm
    ActorPPOTrainer._nashrs_lora_sync_patch = True
