# Документация по использованным сегодня функциям и скриптам (сессия 2026‑01‑09)

Этот документ фиксирует **что именно было запущено/использовано**, какие функции/модули задействованы и какие параметры важны.

## 1) Общая структура проекта и точка входа

### Основные файлы
- **Тренировка**: `scripts/train_flux_tlora.py`
- **Инференс**: `scripts/infer_flux_tlora.py`
- **T‑LoRA реализация/инжект**: `scripts/tlora_flux.py`
- **Конфиг**: `configs/flux_tlora_material.yaml`
- **Данные**: `data/material_brick/images/*.png` + подписи `*.txt`

### Ключевая идея пайплайна
1) Загружаем FLUX пайплайн из diffusers: `FluxPipeline.from_pretrained(...)`.
2) Вставляем (inject) T‑LoRA слои внутрь трансформера: `inject_tlora_into_transformer(...)`.
3) Во время обучения (training) динамически задаём `sigma_mask` в зависимости от timestep и обучаем только параметры T‑LoRA.
4) Во время инференса (inference) прогоняем pipeline, задавая `sigma_mask` на каждом шаге денойзинга (через callback), и/или масштаб T‑LoRA через `_tlora_scale`.

## 2) `scripts/tlora_flux.py` — реализация T‑LoRA

### `TLoRAConfig` (dataclass)
Контейнер параметров адаптера:
- **rank**: максимальный ранг \(R_{max}\)
- **alpha**: LoRA alpha (масштаб)
- **dropout**: dropout в ветке LoRA (для Vanilla‑LoRA)
- **min_rank**: минимальный ранг \(R_{min}\) для ранк‑маскирования
- **alpha_rank_scale**: степень в формуле ранк‑маскирования
- **trainer_type**: `"lora"` или `"ortho_lora"`
- **sig_type**: режим выбора сингулярных значений для `"ortho_lora"` (`principal|last|middle`)

### `sigma_mask_from_timestep(timestep, max_timestep, rank, min_rank, alpha_rank_scale, device, dtype) -> torch.Tensor`
Строит бинарную маску формы `(1, rank)` (1 для активных компонент ранга, 0 для выключенных), где активный ранг вычисляется по формуле:
\[
r = \left\lfloor \left(\frac{T - t}{T}\right)^{\alpha} (R_{max}-R_{min}) \right\rfloor + R_{min}
\]
Используется **в training** (в `train_flux_tlora.py`), где `t` берётся как timestep из noise schedule.

### `sigma_mask_from_step_index(step_idx, num_inference_steps, rank, min_rank, alpha_rank_scale, device, dtype) -> torch.Tensor`
Аналог ранк‑маскирования, но **scheduler‑agnostic**: вместо timestep использует номер шага денойзинга `step_idx` (0..N‑1).
Используется **в inference** (в `infer_flux_tlora.py`) через callback `callback_on_step_end`.

### `TLoRALinear(nn.Module)`
Vanilla вариант T‑LoRA для `nn.Linear`:
- `base`: исходный линейный слой (заморожен)
- `down`: `Linear(in_features -> rank)`
- `up`: `Linear(rank -> out_features)`
- во `forward()` добавляет \(\Delta W\) с учётом `sigma_mask` и масштабов.

**Важно** (фикс этой сессии): владелец‑трансформер хранится не как `nn.Module` атрибут, а как `weakref` в `__dict__`, чтобы избежать циклической регистрации модулей и `RecursionError` при `model.train()`.

### `OrthoTLoRALinear(nn.Module)`
Orthogonal вариант T‑LoRA:
- инициализация через SVD весов базового слоя
- содержит `q_layer`, `p_layer`, `lambda_layer` и буферы `base_*` для нулевого старта дельты
- также использует `sigma_mask`
Имеет тот же фикс owner‑ссылки через `weakref`.

### `_iter_named_linears(root: nn.Module)`
Итератор по всем `nn.Linear` внутри `root` с именами из `named_modules()`.
Используется для поиска целевых слоёв (например `to_q/to_k/to_v`).

### `inject_tlora_into_transformer(transformer, target_module_suffixes, cfg) -> int`
Заменяет найденные `nn.Linear` в `transformer` на `TLoRALinear` или `OrthoTLoRALinear`.
Возвращает количество замен.

Значимые детали:
- сначала собирает список замен, затем делает `setattr(parent, attr, wrapped)` (без мутаций во время обхода).
- вызывает `wrapped.set_owner(transformer)` чтобы адаптер мог читать:
  - `transformer._tlora_sigma_mask`
  - `transformer._tlora_scale`

### `patch_transformer_forward_for_joint_attention_kwargs(transformer) -> None`
Monkey‑patch `transformer.forward`:
- вычитывает `joint_attention_kwargs["sigma_mask"]` и кладёт в `transformer._tlora_sigma_mask`
- вычитывает `joint_attention_kwargs["scale"]` и кладёт в `transformer._tlora_scale`
- фильтрует kwargs по сигнатуре оригинального forward, чтобы не ломаться на “лишних” ключах.

### `tlora_parameters(module) -> List[nn.Parameter]`
Собирает trainable параметры внутри `TLoRALinear/OrthoTLoRALinear`.
Используется для создания оптимизатора (обучаем **только** T‑LoRA веса).

### `save_tlora_weights(module, out_file)` / `load_tlora_weights(module, in_file, strict=False)`
Сохранение/загрузка весов T‑LoRA (обычно `tlora_weights.safetensors`).
Загрузка best‑effort: `strict=False` допускает несовпадения ключей.

## 3) `scripts/train_flux_tlora.py` — тренировка FLUX + T‑LoRA

### CLI
`--config` (обязательный): путь к YAML конфигу  
`--hf_token` (опциональный): токен Hugging Face для gated модели

### PyTorch/SDP compat patch (фикс этой сессии)
В начале файла добавлен shim:
- перехватывает `torch.nn.functional.scaled_dot_product_attention`
- удаляет kwarg `enable_gqa`, если он передан diffusers’ом
Это необходимо, потому что `diffusers` может вызывать SDP с `enable_gqa=...`, а текущий torch может это не поддерживать.

### `CaptionDataset`
Датасет читает:
- изображения из `train_data_dir`
- текстовые подписи из соседних `*.txt` (если отсутствуют — пустая строка)

### `precompute_prompt_cache_cpu_then_to_gpu(pipe, captions, gpu_device, dtype, max_sequence_length=77)`
Оптимизация: кэширует эмбеддинги промптов один раз:
- вызывает `pipe.encode_prompt(...)` на CPU
- переносит эмбеддинги на GPU
- **max_sequence_length** зафиксирован (по умолчанию 77) для согласования с ограничением CLIP‑токенизатора

### Основной цикл `main()`
Ключевые шаги:
- создаётся `Accelerator(mixed_precision, gradient_accumulation_steps)`
- загружается `FluxPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype, token=hf_token)`
- `text_encoder` и `text_encoder_2` остаются на CPU (для снижения VRAM)
- `vae` и `transformer` на GPU
- включается gradient checkpointing (если поддерживается)
- `inject_tlora_into_transformer` + `patch_transformer_forward_for_joint_attention_kwargs`
- строится DataLoader
- кэшируются prompt embeds
- оптимизатор AdamW над `tlora_parameters(transformer)`
- на каждом шаге:
  - кодируем изображение в латенты (`vae.encode`)
  - берём случайный timestep, делаем noisy latents (FlowMatch sigma mixing)
  - pack latents для FLUX‑трансформера
  - вычисляем `sigma_mask_from_timestep` и кладём в `pipe.transformer._tlora_sigma_mask`
  - вызываем `pipe.transformer(...)`
  - MSE loss между предсказанием и noise
  - periodic checkpoint: `save_tlora_weights(.../checkpoint-<step>/tlora_weights.safetensors)`
  - финал: `.../final/tlora_weights.safetensors`

## 4) `scripts/infer_flux_tlora.py` — инференс

### CLI параметры (главные)
- `--config`: YAML (тот же формат, что и training)
- `--lora_path`: путь к папке `checkpoint-XXX` или `final` (или напрямую к `.safetensors`)
- `--prompts_file`: файл с промптами (1 на строку)
- `--outdir`: куда сохранять PNG
- `--num`: сколько изображений **на промпт**
- `--seed`: seed для генератора
- `--steps`: num_inference_steps (если не указан — берётся из конфига или дефолт 25)
- `--lora_scale`: масштаб T‑LoRA (в режиме без `--ab_test`)
- `--ab_test` и `--ab_scales`: генерация нескольких scale‑вариантов на промпт
- `--sequential_offload`: агрессивный CPU‑offload (может быть полезен при нехватке VRAM, но в этой сессии иногда приводил к I/O wait на сетевом диске)

### Как накладывается ранк‑маска на инференсе
Внутри цикла на каждый шаг денойзинга используется callback `callback_on_step_end`:
- вычисляет `sigma_mask_from_step_index(step_idx, ...)`
- кладёт в `joint_attention_kwargs["sigma_mask"]`
- дублирует в `pipe.transformer._tlora_sigma_mask` (best‑effort)

### Выходные файлы
Формат имени:
`single_neutraltrigger_p{p_i:02d}_img{i:02d}_steps{steps}_g{guidance}_lora{scale:.2f}.png`

## 5) Конфиг `configs/flux_tlora_material.yaml`

Ключевые поля:
- `pretrained_model_name_or_path`: `"black-forest-labs/FLUX.1-dev"` (gated)
- `train_data_dir`: `data/material_brick/images`
- `output_dir`: `outputs/flux_tlora_matbrk`
- `mixed_precision`: `"bf16"`
- `max_train_steps`, `checkpointing_steps`, `learning_rate`, и т.д.
- `lora_rank`, `lora_alpha`, `target_modules`
- параметры ранк‑маскирования: `min_rank`, `alpha_rank_scale`, `enable_rank_masking`

## 6) Практические команды, которые использовались

### Установка зависимостей
```bash
cd /workspace/T-Lora
python -m pip install -r requirements.txt
```

### Тренировка
```bash
TOKENIZERS_PARALLELISM=false \
python scripts/train_flux_tlora.py --config configs/flux_tlora_material.yaml
```

### Инференс (10 промптов, 1 картинка на промпт)
```bash
python scripts/infer_flux_tlora.py \
  --config configs/flux_tlora_material.yaml \
  --lora_path outputs/flux_tlora_matbrk/checkpoint-400 \
  --prompts_file outputs/prompts_trzo_b_10.txt \
  --outdir outputs/infer_custom \
  --num 1 \
  --seed 0
```

## 7) Частые проблемы, которые возникали сегодня (и как решались)

### (A) 401 Unauthorized / gated repo
Модель `black-forest-labs/FLUX.1-dev` требует токен и доступ к репозиторию.

### (B) No space left on device
Корневой `/` был ограничен по размеру; HF‑кэш по умолчанию шёл в `/root/.cache`.
Решение: перенести кэши в `/workspace/.cache` через `HF_HOME/HUGGINGFACE_HUB_CACHE` и `TMPDIR`.

### (C) `RecursionError` на `transformer.train()`
Причина: циклическая регистрация модулей из‑за хранения owner‑ссылки (`transformer`) внутри адаптера как `nn.Module` атрибута.
Решение: хранить owner как `weakref` в `__dict__` (фикс в `tlora_flux.py`).

### (D) `scaled_dot_product_attention(... enable_gqa=...)` TypeError
Причина: несовместимость `diffusers` с конкретной сборкой `torch`.
Решение: monkey‑patch SDP, выкидывая `enable_gqa` (фикс в `train_flux_tlora.py` и аналогичный код в `infer_flux_tlora.py` уже присутствовал).

### (E) Инференс может “залипать” в `D` (I/O wait)
Причина: активный offload/подкачка на сетевом диске; при проблемах с I/O процесс может зависнуть.
Практическое решение: перезапуск без `--sequential_offload` (если VRAM позволяет), либо обеспечить быстрый локальный кэш и достаточное место.


