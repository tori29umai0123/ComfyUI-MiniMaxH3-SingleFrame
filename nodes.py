import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
import nodes


CANVAS_MULTIPLE = 32
COMPAT_FRAME_COUNT = 5
FPS = 24
AUDIO_LATENT_FPS = 40


def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _align_frame_count(frame_count):
    frame_count = int(frame_count)
    frame_count = max(COMPAT_FRAME_COUNT, frame_count)
    while frame_count % 17 != 5:
        frame_count += 1
    return frame_count


def _video_latent_t(frame_count):
    return 2 if frame_count <= COMPAT_FRAME_COUNT else ((frame_count - COMPAT_FRAME_COUNT) // 17) * 5 + 2


def _audio_latent_t(frame_count):
    return round((frame_count / FPS) * AUDIO_LATENT_FPS)


def _empty_av_latent(width, height, frame_count, batch_size=1):
    device = comfy.model_management.intermediate_device()
    frame_count = _align_frame_count(frame_count)
    video = torch.zeros([batch_size, 24, _video_latent_t(frame_count), height // 16, width // 16], device=device)
    audio = torch.zeros([batch_size, 32, 2, _audio_latent_t(frame_count)], device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


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
        latent, _ = _empty_av_latent(width, height, COMPAT_FRAME_COUNT)
        return (latent,)


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
                "frame_count": ("INT", {"default": 5, "min": 5, "max": 3600, "step": 17}),
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
            image_mode = frame_count
            frame_count = 5
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
                "frame_count": ("INT", {"default": 5, "min": 5, "max": 3600, "step": 17}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"
    CATEGORY = "model/conditioning/minimax"

    def encode(self, clip, vae, start_frame, end_frame, prompt, width, height, frame_count=5):
        if isinstance(frame_count, str):
            frame_count = 5
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


class MiniMaxH3SelectFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 3600}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select"
    CATEGORY = "image/video"

    def select(self, images, frame_index):
        frame_index = max(0, min(images.shape[0] - 1, frame_index))
        return (images[frame_index:frame_index + 1].clone(),)


NODE_CLASS_MAPPINGS = {
    "EmptyMiniMaxH3SingleFrameLatent": EmptyMiniMaxH3SingleFrameLatent,
    "MiniMaxH3SingleFrameEdit": MiniMaxH3SingleFrameEdit,
    "MiniMaxH3StartEndFrameInterpolate": MiniMaxH3StartEndFrameInterpolate,
    "MiniMaxH3SelectFrame": MiniMaxH3SelectFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyMiniMaxH3SingleFrameLatent": "Empty MiniMax H3 Single Frame Latent",
    "MiniMaxH3SingleFrameEdit": "MiniMax H3 Single Frame Edit",
    "MiniMaxH3StartEndFrameInterpolate": "MiniMax H3 Start End Frame Interpolate",
    "MiniMaxH3SelectFrame": "MiniMax H3 Select Frame",
}
