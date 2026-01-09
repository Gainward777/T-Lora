# История сессии (2026‑01‑09) — T‑LoRA / FLUX

Этот файл — конспект действий и решений из текущего чата.  
**Секреты (HF токен) намеренно не сохраняются**: токен был предоставлен пользователем и применён локально, но здесь он **замаскирован**.

## 1) Цель сессии
- Изучить репозиторий
- Установить зависимости
- Запустить обучение
- Затем запустить инференс на пользовательских промптах и разных чекпоинтах

## 2) Что нашли в репозитории
- В корне есть `requirements.txt`, `README.md`, `configs/flux_tlora_material.yaml`, `data/material_brick/images`.
- Скрипты:
  - `scripts/train_flux_tlora.py`
  - `scripts/infer_flux_tlora.py`
  - `scripts/tlora_flux.py`

## 3) Установка зависимостей
- Выполнено: `python -m pip install -r requirements.txt`
- Проверено окружение:
  - Python 3.11
  - GPU доступна (NVIDIA A40)

## 4) Попытка старта обучения и проблемы доступа
### 4.1 Gated модель FLUX
- Конфиг использует `pretrained_model_name_or_path: "black-forest-labs/FLUX.1-dev"`.
- Первая попытка обучения упала с `401 Unauthorized` (gated repo).
- Пользователь сообщил, что токен есть, но в env‑переменных он не был выставлен.
- Пользователь предоставил HF токен (**замаскировано**: `hf_********************************`).
- Доступ к модели проверен через `huggingface_hub` (успешно).

### 4.2 Нехватка места на `/`
- Скачивание модели в дефолтный кэш `/root/.cache/huggingface` падало с `No space left on device`.
- Решение: перенести HF/transformers/diffusers кэш и `TMPDIR` в `/workspace`, где много места.

## 5) Фиксы кода, сделанные в ходе сессии

### 5.1 `RecursionError` при `transformer.train()`
- Симптом: `RecursionError: maximum recursion depth exceeded` при вызове `.train()` после инжекта T‑LoRA.
- Причина: циклическая регистрация модулей (transformer -> adapter -> transformer).
- Фикс: в `scripts/tlora_flux.py` хранить owner‑ссылку как `weakref` в `__dict__`, чтобы `nn.Module` не регистрировал её как submodule.

### 5.2 `TypeError: scaled_dot_product_attention(... enable_gqa=...)`
- Симптом: падение в training при SDPA с kwarg `enable_gqa`.
- Причина: `diffusers` вызывает SDPA с `enable_gqa`, но текущий `torch` не принимает этот аргумент.
- Фикс: shim/monkey‑patch в `scripts/train_flux_tlora.py`, который выкидывает `enable_gqa` из kwargs.

### 5.3 Стабилизация prompt cache
- В training добавлен `max_sequence_length` (по умолчанию 77) при вычислении prompt embeds, чтобы лучше согласоваться с ограничениями токенизатора CLIP.

## 6) Итог по тренировке
- После фиксов обучение запустилось, в `outputs/flux_tlora_matbrk/` появились чекпоинты (`checkpoint-XXX/tlora_weights.safetensors`) и финальные веса.
- В процессе пользователь просил поток логов; был запущен `tail -f` на текущий лог.

## 7) Правки промптов
- Файл `outputs/prompts_trzo_b_10.txt` использовался как список промптов для инференса.
- По запросу пользователя удалена подстрока `(black terrazzo with white and terracotta chips)` во всех 10 строках.

## 8) Инференс (практика)
- Инференс запускался на:
  - `outputs/flux_tlora_matbrk/checkpoint-300`
  - `outputs/flux_tlora_matbrk/checkpoint-400`
- Наблюдались периодические “залипания” процесса в состоянии `D` (I/O wait) при некоторых режимах offload.
- Для стабильного результата иногда требовался перезапуск без `--sequential_offload`.
- Пример успешного результата: для `checkpoint-400` было сгенерировано 5 PNG в:
  - `outputs/infer_trzo_b_ckpt400_5img_20260109_203224/`

## 9) Важные пути/файлы, которые стоит помнить
- **Конфиг**: `configs/flux_tlora_material.yaml`
- **Данные**: `data/material_brick/images/`
- **Скрипты**: `scripts/train_flux_tlora.py`, `scripts/infer_flux_tlora.py`, `scripts/tlora_flux.py`


