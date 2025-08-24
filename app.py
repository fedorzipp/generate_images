import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import openai
import base64
from PIL import Image
from io import BytesIO

# --- OpenAI API key ---
openai.api_key = "YOUR_API_KEY"  # ⚠️ заміни на свій ключ

# --- Stable Diffusion init ---
@st.cache_resource
def load_sd_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_sd_model()

st.title("🎨 Pipeline: Stable Diffusion → OpenAI API")

prompt = st.text_area("Введіть опис (prompt):", "A fantasy castle on a mountain at sunset")

if st.button("Запустити Pipeline"):
    # Step 1: SD генерує базову картинку
    with st.spinner("Stable Diffusion генерує чорновик..."):
        sd_image = pipe(prompt).images[0]
        st.image(sd_image, caption="Stable Diffusion (чорновик)")

        # Зберігаємо у байти
        buffer = BytesIO()
        sd_image.save(buffer, format="PNG")
        sd_bytes = buffer.getvalue()

    # Step 2: OpenAI робить variation (покращена версія)
    with st.spinner("OpenAI API покращує зображення..."):
        response = openai.images.variations(
            model="gpt-image-1",  # DALL·E 3
            image=sd_bytes,
            size="512x512"
        )
        image_base64 = response.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        openai_image = Image.open(BytesIO(image_bytes))

        st.image(openai_image, caption="OpenAI API (покращена версія)")
