# --- 1. Інсталяція ---

import torch
from diffusers import StableDiffusionPipeline
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "REAL_API_KEY")

# --- 2. OpenAI helper function (V2 API) ---
def openai_edit_from_prompt(prompt):
    """
    Генерує або покращує зображення з Stable Diffusion через новий prompt.
    Використовує тільки V2 API (images.generate).
    """
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024"
    )
    image_data = base64.b64decode(response.data[0].b64_json)
    return Image.open(BytesIO(image_data))

# --- 3. Pipeline function ---
def generate_with_pipeline(prompt, edit_prompt=None, output_sd="sd_output.png", output_final="final_output.png"):
    # 1. Генерація чорновика через SD
    print(f"[SD] Генерація чорновика для prompt: {prompt}")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    sd_image = pipe(prompt).images[0]
    sd_image.save(output_sd)
    print(f"[SD] Збережено чорновик: {output_sd}")

    # 2. Генерація фінального зображення через OpenAI V2
    enhance_prompt = edit_prompt if edit_prompt else prompt
    print(f"[OpenAI] Створення покращеної версії (enhance) з prompt: {enhance_prompt}")
    final_image = openai_edit_from_prompt(enhance_prompt)
    final_image.save(output_final)
    print(f"[OpenAI] Фінальний результат збережено: {output_final}")

    return sd_image, final_image

# --- 4. Тест ---
prompt = "A fantasy castle on a mountain at sunset"
sd_img, final_img = generate_with_pipeline(prompt)
display(sd_img)
display(final_img)

# --- 5. Редагування / зміна стилю ---
prompt = "A futuristic city skyline at night"
edit_prompt = "Make it in cyberpunk neon style"
sd_img, final_img = generate_with_pipeline(prompt, edit_prompt)
display(sd_img)
display(final_img)

