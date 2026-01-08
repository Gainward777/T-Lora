# t-lora_flux
## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Тренировка (Training)


Запуск обучения LoRA производится с помощью конфигурационного файла.

```bash
TOKENIZERS_PARALLELISM=false \
python scripts/train_flux_tlora.py \
  --config configs/flux_tlora_material.yaml
```
Все основные параметры обучения (learning rate, количество шагов, LoRA rank, пути к данным и т.д.) задаются в файле конфигурации:

```bash
configs/flux_tlora_material.yaml
```

Результаты обучения (чекпоинты и финальная LoRA) сохраняются в папку outputs/.

## Инференс (Inference)
После обучения можно выполнить инференс с обученной LoRA:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
python scripts/infer_flux_tlora.py \
  --config configs/flux_tlora_material.yaml \
  --lora_path outputs/flux_tlora_matbrk/final \
  --outdir outputs/infer_ab_final \
  --ab_test \
  --num 2 \
  --seed 0
```

Параметры инференса
--lora_path — путь к обученной LoRA

--outdir — директория для сохранения изображений

--ab_test — генерация A/B сравнения (без LoRA / с LoRA)

--num — количество изображений

--seed — seed для воспроизводимости

Структура проекта
```bash
flux_tlora/
├── scripts/        # скрипты обучения и инференса
├── configs/        # конфигурационные файлы (yaml)
├── data/
│   └── material_brick/
│       └── images/
│           ├── 001.png
│           ├── 001.txt
├── outputs/        # чекпоинты и результаты
├── requirements.txt
└── README.md
```
