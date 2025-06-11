from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.jit
import torch.nn.functional as F

from torch import Tensor
from tqdm import trange

from comfy.utils import ProgressBar
from comfy.model_patcher import ModelPatcher
from comfy.model_base import BaseModel
from comfy.model_management import cast_to_device, get_torch_device
import comfy.utils
import comfy.lora
import folder_paths
import nodes

from . import mat
from .util import (
    gaussian_blur,
    binary_erosion,
    binary_dilation,
    make_odd,
    mask_floor,
    mask_unsqueeze,
    to_torch,
    to_comfy,
    resize_square,
    undo_resize_square,
)

INPAINT_HEAD_TYPE = "INPAINT_HEAD" # Add these lines if not already at top
INPAINT_LORA_TYPE = "INPAINT_LORA"

class InpaintHead(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = torch.nn.Parameter(torch.empty(size=(320, 5, 3, 3), device="cpu"))

    def __call__(self, x):
        x = F.pad(x, (1, 1, 1, 1), "replicate")
        return F.conv2d(x, weight=self.head)


def load_fooocus_patch(lora: dict, to_load: dict):
    patch_dict = {}
    loaded_keys = set()
    for key in to_load.values():
        if value := lora.get(key, None):
            patch_dict[key] = ("fooocus", value)
            loaded_keys.add(key)

    not_loaded = sum(1 for x in lora if x not in loaded_keys)
    if not_loaded > 0:
        print(
            f"[ApplyFooocusInpaint] {len(loaded_keys)} Lora keys loaded, {not_loaded} remaining keys not found in model."
        )
    return patch_dict


if not hasattr(comfy.lora, "calculate_weight") and hasattr(ModelPatcher, "calculate_weight"):
    too_old_msg = "comfyui-inpaint-nodes requires a newer version of ComfyUI (v0.1.1 or later), please update!"
    raise RuntimeError(too_old_msg)


original_calculate_weight = comfy.lora.calculate_weight
injected_model_patcher_calculate_weight = False


def calculate_weight_patched(
    patches, weight, key, intermediate_dtype=torch.float32, original_weights=None
):
    remaining = []

    for p in patches:
        alpha = p[0]
        v = p[1]

        is_fooocus_patch = isinstance(v, tuple) and len(v) == 2 and v[0] == "fooocus"
        if not is_fooocus_patch:
            remaining.append(p)
            continue

        if alpha != 0.0:
            v = v[1]
            w1 = cast_to_device(v[0], weight.device, torch.float32)
            if w1.shape == weight.shape:
                w_min = cast_to_device(v[1], weight.device, torch.float32)
                w_max = cast_to_device(v[2], weight.device, torch.float32)
                w1 = (w1 / 255.0) * (w_max - w_min) + w_min
                weight += alpha * cast_to_device(w1, weight.device, weight.dtype)
            else:
                print(
                    f"[ApplyFooocusInpaint] Shape mismatch {key}, weight not merged ({w1.shape} != {weight.shape})"
                )

    if len(remaining) > 0:
        return original_calculate_weight(remaining, weight, key, intermediate_dtype)
    return weight


def inject_patched_calculate_weight():
    global injected_model_patcher_calculate_weight
    if not injected_model_patcher_calculate_weight:
        print(
            "[comfyui-inpaint-nodes] Injecting patched comfy.model_patcher.ModelPatcher.calculate_weight"
        )
        comfy.lora.calculate_weight = calculate_weight_patched
        injected_model_patcher_calculate_weight = True


class LoadFooocusInpaint: # Renamed to reflect it loads both components separately
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "head_file": (folder_paths.get_filename_list("inpaint"), {"default": "fooocus_inpaint_head.pth"}), # Renamed for clarity
                "lora_file": (folder_paths.get_filename_list("inpaint"), {"default": "fooocus_inpaint_patch.safetensors"}), # Renamed for clarity
            }
        }

    RETURN_TYPES = (INPAINT_HEAD_TYPE, INPAINT_LORA_TYPE) # Return both separately
    RETURN_NAMES = ("INPAINT_HEAD", "INPAINT_LORA") # Names for the outputs
    CATEGORY = "inpaint"
    FUNCTION = "load_models" # Renamed function for clarity

    def load_models(self, head_file: str, lora_file: str):
        # Load Inpaint Head Model
        head_full_path = folder_paths.get_full_path("inpaint", head_file)
        if head_full_path is None:
            raise RuntimeError(f"Head file not found: {head_file}")
        inpaint_head_model = InpaintHead()
        sd_head = torch.load(head_full_path, map_location="cpu", weights_only=True)
        inpaint_head_model.load_state_dict(sd_head)

        # Load Inpaint LoRA Patch
        lora_full_path = folder_paths.get_full_path("inpaint", lora_file)
        if lora_full_path is None:
            raise RuntimeError(f"LoRA file not found: {lora_file}")
        inpaint_lora = comfy.utils.load_torch_file(lora_full_path, safe_load=True)

        return (inpaint_head_model, inpaint_lora)


class ApplyFooocusInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "inpaint_head": (INPAINT_HEAD_TYPE,), # Now directly takes the head model
                "latent": ("LATENT",),
            },
            "optional": { # LoRA is now optional
                "inpaint_lora": (INPAINT_LORA_TYPE, {"default": None}), # Optional input
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "inpaint"
    FUNCTION = "patch"

    _inpaint_head_feature: Tensor | None = None
    _inpaint_block: Tensor | None = None

    def patch(
        self,
        model: ModelPatcher,
        inpaint_head: InpaintHead, # Direct InpaintHead instance
        latent: dict[str, Any],
        inpaint_lora: dict[str, Tensor] | None = None, # Optional LoRA dict
        lora_strength: float = 1.0, # LoRA strength
    ):
        base_model: BaseModel = model.model
        latent_pixels = base_model.process_latent_in(latent["samples"])
        noise_mask = latent["noise_mask"].round()

        latent_mask = F.max_pool2d(noise_mask, (8, 8)).round().to(latent_pixels)

        # Always process and inject the InpaintHead feature, as per your finding
        # that it's crucial for proper fitting, even without LoRA.
        feed = torch.cat([latent_mask, latent_pixels], dim=1)
        inpaint_head.to(device=feed.device, dtype=feed.dtype) # Use inpaint_head directly
        self._inpaint_head_feature = inpaint_head(feed)
        self._inpaint_block = None

        m = model.clone() # Clone the model to apply patches without modifying original

        # Set the input block patch for the InpaintHead feature
        m.set_model_input_block_patch(self._input_block_patch)

        # Conditionally apply LoRA patches if inpaint_lora is provided
        if inpaint_lora is not None:
            lora_keys = comfy.lora.model_lora_keys_unet(model.model, {})
            lora_keys.update({x: x for x in base_model.state_dict().keys()}) # Ensure all keys are considered
            loaded_lora = load_fooocus_patch(inpaint_lora, lora_keys)

            # Apply LoRA with the specified strength
            patched = m.add_patches(loaded_lora, lora_strength)

            not_patched_count = sum(1 for x in loaded_lora if x not in patched)
            if not_patched_count > 0:
                print(f"[ApplyFooocusInpaint] Failed to apply {not_patched_count} LoRA keys")
        else:
            print("[ApplyFooocusInpaint] No Inpaint LoRA provided. Running without LoRA patches.")

        # Ensure the patched calculate_weight is injected (this only happens once)
        inject_patched_calculate_weight()
        return (m,)

    def _input_block_patch(self, h: Tensor, transformer_options: dict):
        # This function remains largely the same, as it's just the mechanism for the head injection
        if transformer_options["block"][1] == 0:
            if self._inpaint_block is None or self._inpaint_block.shape != h.shape:
                assert self._inpaint_head_feature is not None, "InpaintHead feature is None, but _input_block_patch is active."
                batch = h.shape[0] // self._inpaint_head_feature.shape[0]
                self._inpaint_block = self._inpaint_head_feature.to(h).repeat(batch, 1, 1, 1)
            h = h + self._inpaint_block
        return h


class VAEEncodeInpaintConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "pixels": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent_inpaint", "latent_samples")
    FUNCTION = "encode"
    CATEGORY = "inpaint"

    def encode(self, positive, negative, vae, pixels, mask):
        try:
            positive, negative, latent = nodes.InpaintModelConditioning().encode(
                positive, negative, pixels, vae, mask, noise_mask=True
            )
        except TypeError:  # ComfyUI versions older than 2024-11-19
            positive, negative, latent = nodes.InpaintModelConditioning().encode(
                positive, negative, pixels, vae, mask
            )
        latent_inpaint = dict(
            samples=positive[0][1]["concat_latent_image"],
            noise_mask=latent["noise_mask"].round(),
        )
        return (positive, negative, latent_inpaint, latent)


class MaskedFill:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "fill": (["neutral", "telea", "navier-stokes"],),
                "falloff": ("INT", {"default": 0, "min": 0, "max": 8191, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "inpaint"
    FUNCTION = "fill"

    def fill(self, image: Tensor, mask: Tensor, fill: str, falloff: int):
        image = image.detach().clone()
        alpha = mask_unsqueeze(mask_floor(mask))
        assert alpha.shape[0] == image.shape[0], "Image and mask batch size does not match"

        falloff = make_odd(falloff)
        if falloff > 0:
            erosion = binary_erosion(alpha, falloff)
            alpha = alpha * gaussian_blur(erosion, falloff)

        if fill == "neutral":
            m = (1.0 - alpha).squeeze(1)
            for i in range(3):
                image[:, :, :, i] -= 0.5
                image[:, :, :, i] *= m
                image[:, :, :, i] += 0.5
        else:
            import cv2

            method = cv2.INPAINT_TELEA if fill == "telea" else cv2.INPAINT_NS
            for slice, alpha_slice in zip(image, alpha):
                alpha_np = alpha_slice.squeeze().cpu().numpy()
                alpha_bc = alpha_np.reshape(*alpha_np.shape, 1)
                image_np = slice.cpu().numpy()
                filled_np = cv2.inpaint(
                    (255.0 * image_np).astype(np.uint8),
                    (255.0 * alpha_np).astype(np.uint8),
                    3,
                    method,
                )
                filled_np = filled_np.astype(np.float32) / 255.0
                filled_np = image_np * (1.0 - alpha_bc) + filled_np * alpha_bc
                slice.copy_(torch.from_numpy(filled_np))

        return (image,)


class MaskedBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "blur": ("INT", {"default": 255, "min": 3, "max": 8191, "step": 1}),
                "falloff": ("INT", {"default": 0, "min": 0, "max": 8191, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "inpaint"
    FUNCTION = "fill"

    def fill(self, image: Tensor, mask: Tensor, blur: int, falloff: int):
        blur = make_odd(blur)
        falloff = min(make_odd(falloff), blur - 2)
        image, mask = to_torch(image, mask)

        original = image.clone()
        alpha = mask_floor(mask)
        if falloff > 0:
            erosion = binary_erosion(alpha, falloff)
            alpha = alpha * gaussian_blur(erosion, falloff)
        alpha = alpha.expand(-1, 3, -1, -1)

        image = gaussian_blur(image, blur)
        image = original + (image - original) * alpha
        return (to_comfy(image),)


class LoadInpaintModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("inpaint"),),
            }
        }

    RETURN_TYPES = ("INPAINT_MODEL",)
    CATEGORY = "inpaint"
    FUNCTION = "load"

    def load(self, model_name: str):
        from spandrel import ModelLoader

        model_file = folder_paths.get_full_path("inpaint", model_name)
        if model_file is None:
            raise RuntimeError(f"Model file not found: {model_name}")
        if model_file.endswith(".pt"):
            sd = torch.jit.load(model_file, map_location="cpu").state_dict()
        else:
            sd = comfy.utils.load_torch_file(model_file, safe_load=True)

        if "synthesis.first_stage.conv_first.conv.resample_filter" in sd:  # MAT
            model = mat.load(sd)
        else:
            model = ModelLoader().load_from_state_dict(sd)
        model = model.eval()
        return (model,)


class InpaintWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inpaint_model": ("INPAINT_MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "optional_upscale_model": ("UPSCALE_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "inpaint"
    FUNCTION = "inpaint"

    def inpaint(
        self,
        inpaint_model: mat.MAT | Any,
        image: Tensor,
        mask: Tensor,
        seed: int,
        optional_upscale_model=None,
    ):
        if isinstance(inpaint_model, mat.MAT):
            required_size = 512
        elif inpaint_model.architecture.id == "LaMa":
            required_size = 256
        else:
            raise ValueError(f"Unknown model_arch {type(inpaint_model)}")

        if optional_upscale_model != None:
            from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

            upscaler = ImageUpscaleWithModel

        image, mask = to_torch(image, mask)
        batch_size = image.shape[0]
        if mask.shape[0] != batch_size:
            mask = mask[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)

        image_device = image.device
        device = get_torch_device()
        inpaint_model.to(device)
        batch_image = []
        pbar = ProgressBar(batch_size)

        for i in trange(batch_size):
            work_image, work_mask = image[i].unsqueeze(0), mask[i].unsqueeze(0)
            work_image, work_mask, original_size = resize_square(
                work_image, work_mask, required_size
            )
            work_mask = mask_floor(work_mask)

            torch.manual_seed(seed)
            work_image = inpaint_model(work_image.to(device), work_mask.to(device))

            if optional_upscale_model != None:
                work_image = work_image.movedim(1, -1)
                work_image = upscaler.upscale(upscaler, optional_upscale_model, work_image)
                work_image = work_image[0].movedim(-1, 1)

            work_image.to(image_device)
            work_image = undo_resize_square(work_image.to(image_device), original_size)
            work_image = image[i] + (work_image - image[i]) * mask_floor(mask[i])

            batch_image.append(work_image)
            pbar.update(1)

        inpaint_model.cpu()
        result = torch.cat(batch_image, dim=0)
        return (to_comfy(result),)


class DenoiseToCompositingMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "offset": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "threshold": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    CATEGORY = "inpaint"
    FUNCTION = "convert"

    def convert(self, mask: Tensor, offset: float, threshold: float):
        assert 0.0 <= offset < threshold <= 1.0, "Threshold must be higher than offset"
        mask = (mask - offset) * (1 / (threshold - offset))
        mask = mask.clamp(0, 1)
        return (mask,)


class ExpandMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "grow": ("INT", {"default": 16, "min": 0, "max": 8096, "step": 1}),
                "blur": ("INT", {"default": 7, "min": 0, "max": 8096, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    CATEGORY = "inpaint"
    FUNCTION = "expand"

    def expand(self, mask: Tensor, grow: int, blur: int):
        mask = mask_unsqueeze(mask)
        if grow > 0:
            mask = binary_dilation(mask, grow)
        if blur > 0:
            mask = gaussian_blur(mask, make_odd(blur))
        return (mask.squeeze(1),)
