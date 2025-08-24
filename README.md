# Генерація художніх зображень за допомогою Stable Diffusion

## Мета проєкту
Створення художніх зображень на основі текстових описів та їх покращення через модифікацію prompt.

## Інструменти
- Python
- PyTorch
- Diffusers (Stable Diffusion v1.5)
- HuggingFace Hub
- PIL

## Інструкції для запуску
1. Встановити бібліотеки:

pip install diffusers transformers accelerate safetensors pillow
2. Запустити `notebook.ipynb` або `script.py`.
3. Змінити текстовий опис у змінній `prompt`.
4. Генеровані файли зберігаються як `sd_output.png` та `final_output.png`.

## Алгоритм
1. Генерація чорновика з текстового опису через Text-to-Image.
2. Покращення / зміна стилю через Img2Img pipeline.



Встановлення проекту локально:

# Генерація зображень за допомогою Stable Diffusion та OpenAI API

## 🎯 Ціль проєкту
Створити систему для генерації художніх зображень на основі текстових описів за допомогою моделей Stable Diffusion та OpenAI DALL·E.

## 🚀 Запуск

### 1. Встановлення
```bash
git clone https://github.com/fedorzipp/generate_images.git
cd generate_images
pip install -r requirements.txt

Запуск Streamlit-додатку
streamlit run app.py

🚀 Приклади запуску 
python generate_pipeline.py --prompt "A fantasy castle on a mountain" \
    --edit_prompt "Make it in cyberpunk neon style"
