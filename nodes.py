import torch

import comfy.sample
import comfy.samplers
import comfy.ldm.minimax.model as minimax_model
import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.utils
import latent_preview
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


def _shift_target_rope_to_pixel_frame(layout, target_index, strength):
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return

    target_offset = minimax_model.FRAME_RESCALE * int(target_index)
    position_ids = layout.position_ids.clone()
    for kind in ("audio", "video"):
        a, b = next((a, b) for a, b, seg_kind in layout.segments if seg_kind == kind)
        shifted_t = layout.position_ids[a:b, 0] + target_offset
        position_ids[a:b, 0].lerp_(shifted_t, strength)
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
            )
            _freeze_target_video_rope(payload["layout"], frame_index, strength)
            minimax_payload = payload
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass
        return executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    return wrapper


def _target_index_rope_wrapper(target_index, strength):
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
            )
            _shift_target_rope_to_pixel_frame(payload["layout"], target_index, strength)
            minimax_payload = payload
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass
        return executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    return wrapper


def _augment_payload_conditions_like_musubi(payload, video_x, audio_x):
    visual_clean = float(payload.get("visual_cond_noise_aug", minimax_model.VISUAL_COND_TIMESTEP))
    audio_clean = float(payload.get("audio_cond_noise_aug", minimax_model.AUDIO_COND_TIMESTEP))
    if visual_clean >= 1.0 and audio_clean >= 1.0:
        return payload

    generator = torch.Generator(device="cpu").manual_seed(int(payload.get("seed", 0)))
    torch.randn(tuple(video_x.shape), generator=generator, dtype=torch.float32, device="cpu")
    torch.randn(tuple(audio_x.shape), generator=generator, dtype=torch.float32, device="cpu")
    payload = dict(payload)

    if visual_clean < 1.0:
        latents = []
        for latent in payload.get("cond_video_latents", []):
            noise = torch.randn(tuple(latent.shape), generator=generator, dtype=torch.float32, device="cpu")
            noise = noise.to(device=latent.device, dtype=latent.dtype)
            latents.append(visual_clean * latent + (1.0 - visual_clean) * noise)
        payload["cond_video_latents"] = latents
        payload["visual_cond_noise_aug"] = 1.0

    if audio_clean < 1.0:
        latents = []
        for latent in payload.get("cond_audio_latents", []):
            noise = torch.randn(tuple(latent.shape), generator=generator, dtype=torch.float32, device="cpu")
            noise = noise.to(device=latent.device, dtype=latent.dtype)
            latents.append(audio_clean * latent + (1.0 - audio_clean) * noise)
        payload["cond_audio_latents"] = latents
        payload["audio_cond_noise_aug"] = 1.0

    return payload


def _musubi_one_frame_wrapper(target_index, strength):
    def wrapper(executor, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        try:
            video_x, audio_x = x[0], x[1]
            latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
            lat_h = (lat_h + 1) // 2 * 2
            lat_w = (lat_w + 1) // 2 * 2
            payload = _augment_payload_conditions_like_musubi(dict(minimax_payload or {}), video_x, audio_x)
            payload["layout"] = minimax_model.PackedLayout(
                context.shape[1],
                latent_t,
                lat_h,
                lat_w,
                audio_x.shape[-1],
                keyframes=payload.get("keyframes"),
                refs=payload.get("refs"),
            )
            _shift_target_rope_to_pixel_frame(payload["layout"], target_index, strength)
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


def _musubi_h3_video_sigmas(steps, video_shift):
    base = torch.linspace(1.0, 0.0, int(steps) + 1, dtype=torch.float64)
    shift = float(video_shift)
    return (shift * base / (1.0 + (shift - 1.0) * base)).to(torch.float32)


@torch.no_grad()
def _musubi_h3_euler_sampler_function(model, x, sigmas, extra_args=None, callback=None, disable=None):
    # Copied in spirit from musubi_tuner.minimax_h3.sampling.sample_joint_av:
    # the model predicts the dataward velocity v, and the latent update is
    # x_next = x + (sigma_i - sigma_next) * v. Comfy's wrapper exposes x0_hat,
    # so recover v = (x0_hat - x) / sigma for the packed MiniMax H3 latent.
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        denoised = model(x, sigma * s_in, **extra_args)
        velocity = (denoised - x) / sigma.clamp(min=1e-6).to(x)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
        x = x + (sigma - sigmas[i + 1]).to(x) * velocity
    return x


def _sample_with_musubi_euler(model, seed, steps, cfg, positive, negative, latent, sigmas, disable_noise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )

    if disable_noise:
        noise = comfy.sample.prepare_empty_noise(latent_image)
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    sampler = comfy.samplers.KSAMPLER(_musubi_h3_euler_sampler_function)
    noise_mask = latent.get("noise_mask", None)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample_custom(
        model,
        noise,
        cfg,
        sampler,
        sigmas,
        positive,
        negative,
        latent_image,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return (out,)


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


class MiniMaxH3TargetIndexRoPEPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "target_index": ("INT", {
                    "default": 24,
                    "min": -3600,
                    "max": 3600,
                    "tooltip": "Pixel-frame index for the generated target frame. Use 24 to match FL2VA one-frame edit LoRA training with fp_1f_target_index=24.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How strongly to move the target video token RoPE time to target_index. Use 1.0 to match the training index exactly.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patches/minimax"

    def patch(self, model, target_index, strength):
        m = model.clone()
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "minimax_h3_target_index_rope",
            _target_index_rope_wrapper(target_index, strength),
        )
        return (m,)


class MiniMaxH3MusubiEulerOneFrameSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MiniMax H3 model, optionally already loaded with LoRA."}),
                "seed": ("INT", {
                    "default": 1234,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Noise seed. Video noise is drawn first, audio noise second, matching the MiniMax H3 packed latent order.",
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of denoising steps. This uses Musubi's shifted MiniMax H3 sigma schedule.",
                }),
                "cfg": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Classifier-free guidance scale. Use 1.0 to match the provided Musubi command.",
                }),
                "h3_shift_video": ("FLOAT", {
                    "default": 12.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "MiniMax H3 video sigma shift. Musubi default is 12.0.",
                }),
                "h3_shift_audio": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "MiniMax H3 audio sigma shift. Musubi default is 3.0.",
                }),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "target_index": ("INT", {
                    "default": 24,
                    "min": -3600,
                    "max": 3600,
                    "tooltip": "Generated target pixel-frame index. Use 24 for --one_frame target_index=24.",
                }),
                "target_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Strength of the target-index RoPE shift. Use 1.0 to match Musubi.",
                }),
            },
            "optional": {
                "disable_noise": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "model/sampling/minimax"

    def sample(self, model, seed, steps, cfg, h3_shift_video, h3_shift_audio, positive, negative, latent_image, target_index, target_strength, disable_noise=False):
        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})
        transformer_options["minimax_h3_sigma_shift_video"] = float(h3_shift_video)
        transformer_options["minimax_h3_sigma_shift_audio"] = float(h3_shift_audio)
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "minimax_h3_musubi_euler_target_index_rope",
            _musubi_one_frame_wrapper(target_index, target_strength),
        )
        sigmas = _musubi_h3_video_sigmas(steps, h3_shift_video)
        return _sample_with_musubi_euler(m, seed, steps, cfg, positive, negative, latent_image, sigmas, disable_noise=disable_noise)


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


def _decode_minimax_single_latent_temporal_first_frame(vae, latent):
    if len(latent.shape) != 5 or latent.shape[2] != 1:
        return None

    model = getattr(vae, "first_stage_model", None)
    if not all(hasattr(model, attr) for attr in ("decode_temporal", "latents_mean", "latents_std", "pixel_mean", "pixel_std")):
        return None

    with comfy.model_management.cuda_device_context(vae.device):
        decode_shape = list(latent.shape)
        decode_shape[2] = 2
        memory_used = vae.memory_used_decode(tuple(decode_shape), vae.vae_dtype)
        comfy.model_management.load_models_gpu(
            [vae.patcher],
            memory_required=memory_used,
            force_full_load=vae.disable_offload,
        )
        z = latent.to(device=vae.device, dtype=vae.vae_dtype)
        latents_mean = model.latents_mean.view(1, -1, 1, 1, 1).to(z)
        latents_std = model.latents_std.view(1, -1, 1, 1, 1).to(z)
        z = z * latents_std + latents_mean
        z = torch.cat([z, z[:, :, -1:]], dim=2)

        frames = model.decode_temporal(z).float()
        frames = frames[:, :, :1, :, :]
        frames = frames.to(device=vae.output_device, dtype=vae.vae_output_dtype(), copy=True)

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
                    "tooltip": "Select the decoded video frame. MiniMax H3 single-frame latents are duplicated only for temporal VAE decode, then the first frame is kept.",
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

        images = _decode_minimax_single_latent_temporal_first_frame(vae, latent)
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
                "control_index": ("INT", {
                    "default": 0,
                    "min": -3600,
                    "max": 3600,
                    "tooltip": "Pixel-frame index for the source control image. Use 0 to match --one_frame control_index=0.",
                }),
                "visual_cond_clean": ("FLOAT", {
                    "default": 0.999,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "MiniMax H3 visual condition clean coefficient. Musubi default is 0.999.",
                }),
                "audio_cond_clean": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "MiniMax H3 audio condition clean coefficient. Musubi default is 1.0.",
                }),
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

    def encode(self, clip, vae, prompt, width, height, frame_count=5, control_index=0, visual_cond_clean=0.999, audio_cond_clean=1.0, image_mode="keyframe", image=None):
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
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_refs": [ref],
                "minimax_visual_cond_noise_aug": float(visual_cond_clean),
                "minimax_audio_cond_noise_aug": float(audio_cond_clean),
            })
        else:
            tokens = clip.tokenize(prompt, images=[img])
            cond = clip.encode_from_tokens_scheduled(tokens)
            keyframe = {
                "resolved_frame_index": int(control_index),
                "latent": vae.encode(img),
            }
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": [keyframe],
                "minimax_frame_count": frame_count,
                "minimax_visual_cond_noise_aug": float(visual_cond_clean),
                "minimax_audio_cond_noise_aug": float(audio_cond_clean),
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
    "MiniMaxH3TargetIndexRoPEPatch": MiniMaxH3TargetIndexRoPEPatch,
    "MiniMaxH3MusubiEulerOneFrameSampler": MiniMaxH3MusubiEulerOneFrameSampler,
    "MiniMaxH3VAEDecodeFrame": MiniMaxH3VAEDecodeFrame,
    "MiniMaxH3SingleFrameEdit": MiniMaxH3SingleFrameEdit,
    "MiniMaxH3StartEndFrameInterpolate": MiniMaxH3StartEndFrameInterpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyMiniMaxH3SingleFrameLatent": "Empty MiniMax H3 Single Frame Latent",
    "MiniMaxH3TemporalRoPEPatch": "MiniMax H3 Temporal RoPE Patch",
    "MiniMaxH3TargetIndexRoPEPatch": "MiniMax H3 Target Index RoPE Patch",
    "MiniMaxH3MusubiEulerOneFrameSampler": "MiniMax H3 Musubi One Frame Sampler",
    "MiniMaxH3VAEDecodeFrame": "MiniMax H3 VAE Decode Frame",
    "MiniMaxH3SingleFrameEdit": "MiniMax H3 Single Frame Edit",
    "MiniMaxH3StartEndFrameInterpolate": "MiniMax H3 Start End Frame Interpolate",
}
