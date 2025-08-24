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
