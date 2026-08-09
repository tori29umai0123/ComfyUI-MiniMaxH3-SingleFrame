import torch

import comfy.ldm.minimax.model as minimax_model
import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.utils
import node_helpers
import nodes


CANVAS_MULTIPLE = 32
COMPAT_FRAME_COUNT = 5
SINGLE_FRAME_COUNT = 1
FPS = 24
AUDIO_LATENT_FPS = 40
MAX_FRAME_COUNT = 3600
COMPAT_FRAME_COUNT_OPTIONS = [str(i) for i in range(COMPAT_FRAME_COUNT, MAX_FRAME_COUNT + 1, 17)]
SINGLE_FRAME_COUNT_OPTIONS = [str(SINGLE_FRAME_COUNT)] + COMPAT_FRAME_COUNT_OPTIONS


def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _align_frame_count(frame_count):
    frame_count = int(frame_count)
    if frame_count <= SINGLE_FRAME_COUNT:
        return SINGLE_FRAME_COUNT
    frame_count = max(COMPAT_FRAME_COUNT, frame_count)
    while frame_count % 17 != 5:
        frame_count += 1
    return frame_count


def _parse_frame_count(frame_count, default):
    try:
        return int(frame_count)
    except (TypeError, ValueError):
        return default


def _video_latent_t(frame_count):
    if frame_count <= SINGLE_FRAME_COUNT:
        return 1
    return 2 if frame_count <= COMPAT_FRAME_COUNT else ((frame_count - COMPAT_FRAME_COUNT) // 17) * 5 + 2


def _audio_latent_t(frame_count):
    return round((frame_count / FPS) * AUDIO_LATENT_FPS)


def _empty_av_latent(width, height, frame_count, batch_size=1):
    device = comfy.model_management.intermediate_device()
    frame_count = _align_frame_count(frame_count)
    video = torch.zeros([batch_size, 24, _video_latent_t(frame_count), height // 16, width // 16], device=device)
    audio = torch.zeros([batch_size, 32, 2, _audio_latent_t(frame_count)], device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _freeze_target_video_rope(layout, frame_index, strength):
    video_seg = next((a, b) for a, b, kind in layout.segments if kind == "video")
    a, b = video_seg
    latent_t = layout.signature[1]
    frame_rows = (b - a) // latent_t
    frame_index = max(0, min(latent_t - 1, int(frame_index)))
    strength = max(0.0, min(1.0, float(strength)))
    ref_t = layout.position_ids[a + frame_index * frame_rows, 0]
    position_ids = layout.position_ids.clone()
    position_ids[a:b, 0].lerp_(ref_t, strength)
    layout.position_ids = position_ids


def _temporal_rope_wrapper(frame_index, strength):
    def wrapper(executor, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        try:
            video_x, audio_x = x[0], x[1]
            latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
            lat_h = (lat_h + 1) // 2 * 2
            lat_w = (lat_w + 1) // 2 * 2
            payload = dict(minimax_payload or {})
            payload["layout"] = minimax_model.PackedLayout(
                context.shape[1],
                latent_t,
                lat_h,
                lat_w,
                audio_x.shape[-1],
                keyframes=payload.get("keyframes"),
                refs=payload.get("refs"),
                frame_count=payload.get("frame_count"),
            )
            _freeze_target_video_rope(payload["layout"], frame_index, strength)
            minimax_payload = payload
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass
        return executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    return wrapper


def _snap_canvas(width, height):
    return (
        max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


class EmptyMiniMaxH3SingleFrameLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "model/latent/minimax"

    def generate(self, width, height):
        width, height = _snap_canvas(width, height)
        latent, _ = _empty_av_latent(width, height, SINGLE_FRAME_COUNT)
        return (latent,)


class MiniMaxH3TemporalRoPEPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 3600}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patches/minimax"

    def patch(self, model, frame_index, strength):
        m = model.clone()
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "minimax_h3_temporal_rope",
            _temporal_rope_wrapper(frame_index, strength),
        )
        return (m,)


def _select_decoded_frames(images, frame_select, frame_index):
    if len(images.shape) != 5:
        return images

    if frame_select == "all_frames":
        return images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

    frame_count = images.shape[1]
    if frame_select == "first":
        index = 0
    elif frame_select == "middle":
        index = frame_count // 2
    elif frame_select == "last":
        index = frame_count - 1
    else:
        index = max(0, min(frame_count - 1, int(frame_index)))

    return images[:, index]


def _decode_minimax_single_latent_first_frame(vae, latent):
    if len(latent.shape) != 5 or latent.shape[2] != 1:
        return None

    model = getattr(vae, "first_stage_model", None)
    if not all(hasattr(model, attr) for attr in ("_adaptive_decode", "latents_mean", "latents_std", "pixel_mean", "pixel_std")):
        return None

    with comfy.model_management.cuda_device_context(vae.device):
        memory_used = vae.memory_used_decode(latent.shape, vae.vae_dtype)
        comfy.model_management.load_models_gpu(
            [vae.patcher],
            memory_required=memory_used,
            force_full_load=vae.disable_offload,
        )
        z = latent.to(device=vae.device, dtype=vae.vae_dtype)
        latents_mean = model.latents_mean.view(1, -1, 1, 1, 1).to(z)
        latents_std = model.latents_std.view(1, -1, 1, 1, 1).to(z)
        z = z * latents_std + latents_mean

        frames = model._adaptive_decode(z).float()
        frames = frames[:, :, :1, :, :]
        frames.mul_(model.pixel_std.to(frames)).add_(model.pixel_mean.to(frames)).clamp_(0.0, 1.0).mul_(2.0).sub_(1.0)
        frames = frames.to(device=vae.output_device, dtype=vae.vae_output_dtype(), copy=True)
        vae.process_output(frames)

    return frames.to(vae.output_device).movedim(1, -1)


class MiniMaxH3VAEDecodeFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "frame_select": (["first", "middle", "last", "index", "all_frames"], {
                    "default": "first",
                    "tooltip": "Select the decoded video frame. MiniMax H3 single-frame latents keep the first internal VAE frame.",
                }),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 3600, "tooltip": "Used only when frame_select is index."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "model/latent/minimax"

    def decode(self, vae, samples, frame_select="last", frame_index=0):
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]

        images = _decode_minimax_single_latent_first_frame(vae, latent)
        if images is None:
            images = vae.decode(latent)
        images = _select_decoded_frames(images, frame_select, frame_index)
        return (images,)


class MiniMaxH3SingleFrameEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "frame_count": (SINGLE_FRAME_COUNT_OPTIONS, {"default": str(SINGLE_FRAME_COUNT)}),
                "image_mode": (["keyframe", "reference"], {
                    "tooltip": "keyframe anchors the source image at frame 0; reference adds it as <Picture 1> conditioning.",
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"
    CATEGORY = "model/conditioning/minimax"

    def encode(self, clip, vae, prompt, width, height, frame_count=5, image_mode="keyframe", image=None):
        if isinstance(frame_count, str):
            parsed_frame_count = _parse_frame_count(frame_count, None)
            if parsed_frame_count is None:
                image_mode = frame_count
                frame_count = 5
            else:
                frame_count = parsed_frame_count
        width, height = _snap_canvas(width, height)
        latent, frame_count = _empty_av_latent(width, height, frame_count)

        if image is None:
            tokens = clip.tokenize(prompt)
            return (clip.encode_from_tokens_scheduled(tokens), latent)

        img = _resize(image[:1], width, height, "disabled")

        if image_mode == "reference":
            tokens = clip.tokenize(prompt, minimax_ref_items=[{"type": "image", "data": img}])
            cond = clip.encode_from_tokens_scheduled(tokens)
            ref = {
                "kind": "image",
                "latent_h": height // 16,
                "latent_w": width // 16,
                "latent": vae.encode(img),
            }
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": [ref]})
        else:
            tokens = clip.tokenize(prompt, images=[img])
            cond = clip.encode_from_tokens_scheduled(tokens)
            keyframe = {
                "resolved_frame_index": 0,
                "latent": vae.encode(img),
            }
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": [keyframe],
                "minimax_frame_count": frame_count,
            })

        return (cond, latent)


class MiniMaxH3StartEndFrameInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "start_frame": ("IMAGE",),
                "end_frame": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "frame_count": (COMPAT_FRAME_COUNT_OPTIONS, {"default": str(COMPAT_FRAME_COUNT)}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"
    CATEGORY = "model/conditioning/minimax"

    def encode(self, clip, vae, start_frame, end_frame, prompt, width, height, frame_count=5):
        frame_count = _parse_frame_count(frame_count, COMPAT_FRAME_COUNT)
        width, height = _snap_canvas(width, height)
        latent, frame_count = _empty_av_latent(width, height, frame_count)

        start_img = _resize(start_frame[:1], width, height, "disabled")
        end_img = _resize(end_frame[:1], width, height, "disabled")

        tokens = clip.tokenize(prompt, images=[start_img, end_img])
        cond = clip.encode_from_tokens_scheduled(tokens)
        keyframes = [
            {
                "resolved_frame_index": 0,
                "latent": vae.encode(start_img),
            },
            {
                "resolved_frame_index": frame_count - 1,
                "latent": vae.encode(end_img),
            },
        ]
        cond = node_helpers.conditioning_set_values(cond, {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
        })
        return (cond, latent)


NODE_CLASS_MAPPINGS = {
    "EmptyMiniMaxH3SingleFrameLatent": EmptyMiniMaxH3SingleFrameLatent,
    "MiniMaxH3TemporalRoPEPatch": MiniMaxH3TemporalRoPEPatch,
    "MiniMaxH3VAEDecodeFrame": MiniMaxH3VAEDecodeFrame,
    "MiniMaxH3SingleFrameEdit": MiniMaxH3SingleFrameEdit,
    "MiniMaxH3StartEndFrameInterpolate": MiniMaxH3StartEndFrameInterpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyMiniMaxH3SingleFrameLatent": "Empty MiniMax H3 Single Frame Latent",
    "MiniMaxH3TemporalRoPEPatch": "MiniMax H3 Temporal RoPE Patch",
    "MiniMaxH3VAEDecodeFrame": "MiniMax H3 VAE Decode Frame",
    "MiniMaxH3SingleFrameEdit": "MiniMax H3 Single Frame Edit",
    "MiniMaxH3StartEndFrameInterpolate": "MiniMax H3 Start End Frame Interpolate",
}
