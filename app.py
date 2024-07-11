"""
Modified from https://github.com/lllyasviel/Paints-UNDO/blob/main/gradio_app.py
"""

import functools

import gradio as gr
import numpy as np
import cv2
import torch

from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from imgutils.metrics import lpips_difference
from imgutils.tagging import get_wd14_tags

from diffusers_helper.code_cond import unet_add_coded_conds
from diffusers_helper.cat_cond import unet_add_concat_conds
from diffusers_helper.k_diffusion import KDiffusionSampler
from diffusers_helper.attention import AttnProcessor2_0_xformers, XFORMERS_AVAIL

from lineart_models import MangaLineExtraction, LineartAnimeDetector, LineartDetector


def resize_and_center_crop(
    image, target_width, target_height=None, interpolation=cv2.INTER_AREA
):
    original_height, original_width = image.shape[:2]
    if target_height is None:
        aspect_ratio = original_width / original_height
        target_pixel_count = target_width * target_width
        target_height = (target_pixel_count / aspect_ratio) ** 0.5
        target_width = target_height * aspect_ratio
    target_height = int(target_height)
    target_width = int(target_width)
    print(
        f"original_height={original_height}, "
        f"original_width={original_width}, "
        f"target_height={target_height}, "
        f"target_width={target_width}"
    )
    k = max(target_height / original_height, target_width / original_width)
    new_width = int(round(original_width * k))
    new_height = int(round(original_height * k))
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=interpolation
    )
    x_start = (new_width - target_width) // 2
    y_start = (new_height - target_height) // 2
    cropped_image = resized_image[
        y_start : y_start + target_height, x_start : x_start + target_width
    ]
    return cropped_image


class ModifiedUNet(UNet2DConditionModel):
    @classmethod
    def from_config(cls, *args, **kwargs):
        m = super().from_config(*args, **kwargs)
        unet_add_concat_conds(unet=m, new_channels=4)
        unet_add_coded_conds(unet=m, added_number_count=1)
        return m


DEVICE = "cuda"
torch._dynamo.config.cache_size_limit = 256


lineart_models = []

lineart_model = MangaLineExtraction("cuda", "./hf_download")
lineart_model.load_model()
lineart_model.model.to(device=DEVICE).eval()
lineart_models.append(lineart_model)

lineart_model = LineartAnimeDetector()
lineart_model.model.to(device=DEVICE).eval()
lineart_models.append(lineart_model)

lineart_model = LineartDetector()
lineart_model.model.to(device=DEVICE).eval()
lineart_models.append(lineart_model)


model_name = "lllyasviel/paints_undo_single_frame"
tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
    model_name, subfolder="tokenizer"
)
text_encoder: CLIPTextModel = (
    CLIPTextModel.from_pretrained(
        model_name,
        subfolder="text_encoder",
    )
    .to(dtype=torch.float16, device=DEVICE)
    .eval()
)
vae: AutoencoderKL = (
    AutoencoderKL.from_pretrained(
        model_name,
        subfolder="vae",
    )
    .to(dtype=torch.bfloat16, device=DEVICE)
    .eval()
)
unet: ModifiedUNet = (
    ModifiedUNet.from_pretrained(
        model_name,
        subfolder="unet",
    )
    .to(dtype=torch.float16, device=DEVICE)
    .eval()
)

if XFORMERS_AVAIL:
    unet.set_attn_processor(AttnProcessor2_0_xformers())
    vae.set_attn_processor(AttnProcessor2_0_xformers())
else:
    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())

# text_encoder = torch.compile(text_encoder, backend="eager", dynamic=True)
# vae = torch.compile(vae, backend="eager", dynamic=True)
# unet = torch.compile(unet, mode="reduce-overhead", dynamic=True)
# for model in lineart_models:
#     model.model = torch.compile(model.model, backend="eager", dynamic=True)
k_sampler = KDiffusionSampler(
    unet=unet,
    timesteps=1000,
    linear_start=0.00085,
    linear_end=0.020,
    linear=True,
)


@torch.inference_mode()
def encode_cropped_prompt_77tokens(txt: str):
    cond_ids = tokenizer(
        txt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device=text_encoder.device)
    text_cond = text_encoder(cond_ids, attention_mask=None).last_hidden_state
    return text_cond


@torch.inference_mode()
def encode_cropped_prompt(txt: str, max_length=150):
    cond_ids = tokenizer(
        txt,
        padding="max_length",
        max_length=max_length + 2,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device=text_encoder.device)
    if max_length + 2 > tokenizer.model_max_length:
        input_ids = cond_ids.squeeze(0)
        id_list = list(
            range(
                1,
                max_length + 2 - tokenizer.model_max_length + 2,
                tokenizer.model_max_length - 2,
            )
        )
        text_cond_list = []
        for i in id_list:
            ids_chunk = (
                input_ids[0].unsqueeze(0),
                input_ids[i : i + tokenizer.model_max_length - 2],
                input_ids[-1].unsqueeze(0),
            )
            if torch.all(ids_chunk[1] == tokenizer.pad_token_id):
                break
            text_cond = text_encoder(
                torch.concat(ids_chunk).unsqueeze(0)
            ).last_hidden_state
            if text_cond_list == []:
                text_cond_list.append(text_cond[:, :1])
            text_cond_list.append(text_cond[:, 1 : tokenizer.model_max_length - 1])
        text_cond_list.append(text_cond[:, -1:])
        text_cond = torch.concat(text_cond_list, dim=1)
    else:
        text_cond = text_encoder(cond_ids, attention_mask=None).last_hidden_state
    return text_cond.flatten(0, 1).unsqueeze(0)


@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h


@torch.inference_mode()
def interrogator_process(x):
    img = Image.fromarray(x)
    rating, features, chars = get_wd14_tags(
        img, general_threshold=0.3, character_threshold=0.75, no_underline=True
    )
    result = ""
    for char in chars:
        result += char
        result += ", "
    for feature in features:
        result += feature
        result += ", "
    result += max(rating, key=rating.get)
    return result, f'{len(tokenizer.tokenize(result))}'


@torch.inference_mode()
def process(
    input_fg,
    prompt,
    input_undo_steps,
    image_width,
    seed,
    steps,
    n_prompt,
    cfg,
    num_sets,
    progress=gr.Progress(),
):
    lineart_fg = input_fg
    linearts = []
    for model in lineart_models:
        linearts.append(model(lineart_fg))
    fg = resize_and_center_crop(input_fg, image_width)
    for i, lineart in enumerate(linearts):
        lineart = resize_and_center_crop(lineart, fg.shape[1], fg.shape[0])
        linearts[i] = lineart

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = (
        vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    )

    conds = encode_cropped_prompt(prompt)
    unconds = encode_cropped_prompt_77tokens(n_prompt)
    print(conds.shape, unconds.shape)
    torch.cuda.empty_cache()

    fs = torch.tensor(input_undo_steps).to(device=unet.device, dtype=torch.long)
    initial_latents = torch.zeros_like(concat_conds)
    concat_conds = concat_conds.to(device=unet.device, dtype=unet.dtype)
    latents = []
    rng = torch.Generator(device=DEVICE).manual_seed(int(seed))
    latents = (
        k_sampler(
            initial_latent=initial_latents,
            strength=1.0,
            num_inference_steps=steps,
            guidance_scale=cfg,
            batch_size=len(input_undo_steps) * num_sets,
            generator=rng,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            cross_attention_kwargs={
                "concat_conds": concat_conds,
                "coded_conds": fs,
            },
            same_noise_in_batch=False,
            progress_tqdm=functools.partial(
                progress.tqdm, desc="Generating Key Frames"
            ),
        ).to(vae.dtype)
        / vae.config.scaling_factor
    )
    torch.cuda.empty_cache()

    pixels = torch.concat(
        [vae.decode(latent.unsqueeze(0)).sample for latent in latents]
    )
    pixels = pytorch2numpy(pixels)
    pixels_with_lpips = []
    lineart_pils = [Image.fromarray(lineart) for lineart in linearts]
    for pixel in pixels:
        pixel_pil = Image.fromarray(pixel)
        pixels_with_lpips.append(
            (
                sum(
                    [
                        lpips_difference(lineart_pil, pixel_pil)
                        for lineart_pil in lineart_pils
                    ]
                ),
                pixel,
            )
        )
    pixels = np.stack(
        [i[1] for i in sorted(pixels_with_lpips, key=lambda x: x[0])], axis=0
    )
    torch.cuda.empty_cache()

    return pixels, np.stack(linearts)


block = gr.Blocks().queue()
with block:
    gr.Markdown("# Sketch/Lineart extractor")

    with gr.Row():
        with gr.Column():
            input_fg = gr.Image(
                sources=["upload"], type="numpy", label="Image", height=384
            )
            with gr.Row():
                with gr.Column(scale=5):
                    prompt = gr.Textbox(label="Output Prompt", interactive=True)
                    n_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="lowres, worst quality, bad anatomy, bad hands, text, extra digit, fewer digits, cropped, low quality, jpeg artifacts, signature, watermark, username",
                    )
                    input_undo_steps = gr.Dropdown(
                        label="Operation Steps",
                        value=[900, 925, 950, 975],
                        choices=list(range(0, 1000, 5)),
                        multiselect=True,
                    )
                    num_sets = gr.Slider(
                        label="Num Sets", minimum=1, maximum=10, value=3, step=1
                    )
                with gr.Column(scale=2, min_width=160):
                    token_counter = gr.Textbox(label="Tokens Count", lines=1, interactive=False)
                    recaption_button = gr.Button(value="Tagging", interactive=True)
                    seed = gr.Slider(
                        label="Seed", minimum=0, maximum=50000, step=1, value=37462
                    )
                    image_width = gr.Slider(
                        label="Target size", minimum=512, maximum=1024, value=768, step=32
                    )
                    steps = gr.Slider(
                        label="Steps", minimum=1, maximum=32, value=16, step=1
                    )
                    cfg = gr.Slider(
                        label="CFG Scale", minimum=1.0, maximum=16, value=5, step=0.05
                    )

        with gr.Column():
            key_gen_button = gr.Button(value="Generate Sketch", interactive=False)
            gr.Markdown("#### Sketch Outputs")
            result_gallery = gr.Gallery(
                height=384, object_fit="contain", label="Sketch Outputs", columns=4
            )
            gr.Markdown("#### Line Art Outputs")
            lineart_result = gr.Gallery(
                height=384,
                object_fit="contain",
                label="LineArt outputs",
            )

    input_fg.change(
        lambda x: [
            *(interrogator_process(x) if x is not None else ("", "")),
            gr.update(interactive=True),
        ],
        inputs=[input_fg],
        outputs=[prompt, token_counter, key_gen_button],
    )
    recaption_button.click(
        lambda x: [
            *(interrogator_process(x) if x is not None else ("", "")),
            gr.update(interactive=True),
        ],
        inputs=[input_fg],
        outputs=[prompt, token_counter, key_gen_button],
    )

    key_gen_button.click(
        fn=process,
        inputs=[
            input_fg,
            prompt,
            input_undo_steps,
            image_width,
            seed,
            steps,
            n_prompt,
            cfg,
            num_sets,
        ],
        outputs=[result_gallery, lineart_result],
    ).then(
        lambda: gr.update(interactive=True),
        outputs=[key_gen_button],
    )

block.queue().launch()
