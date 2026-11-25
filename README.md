# mlops_project_pro_max_alpha_v2
Проект по дисциплине "Промышленная разработка ML"

Поиск похожих товаров по фото (Stanford Online Products): клиент считает эмбеддинг
моделью с Hugging Face, сервер NVIDIA Triton (Python backend + FAISS) возвращает
ID 5 ближайших фото каталога.

## Окружение и версии

| Компонент | Версия | Где зафиксировано |
|---|---|---|
| Python (клиент, скрипты) | 3.12 (`>=3.12,<3.13`) | `pyproject.toml`, `.python-version` |
| Менеджер зависимостей | `uv` (проверено на 0.13.1) | `uv.lock` |
| Triton Server | образ `nvcr.io/nvidia/tritonserver:26.11-py3` (Triton 2.75.0) | этот README, далее `docker/Dockerfile` |
| Digest образа (linux/arm64) | `sha256:cdd351866b6dfbbae6abefee8bab47f2808cd0be33b6fb7825c6f591e5cb11e9` | этот README |
| Python внутри образа | 3.12.3 (Ubuntu 24.04.5 LTS), предустановлен numpy 1.26.4 | проверено `docker run` |
| `tritonclient[http]` | 2.75.0 (совпадает с версией сервера) | `pyproject.toml` |
| `torch` / `transformers` | 2.15.0 / 5.24.1 | `uv.lock` |
| `faiss-cpu` / `numpy` / `pillow` | 1.16.0 / 2.5.5 / 13.0.0 | `uv.lock` |

Машина разработки: macOS 26.5, Apple Silicon (arm64), без GPU.
`uv.lock` фиксирует одинаковые версии для всех платформ. Wheel `torch 2.15.0`
для Mac собран под `macosx_14_0_arm64`, поэтому нужна **macOS ≥ 14**.

### Почему так

- **Python 3.12.** Для 3.12 и 3.13 `uv pip compile` подобрал одинаковые версии
  `torch`, `faiss-cpu`, `transformers`, `tritonclient` (и на macOS arm64, и на Linux x86_64).
  Выбрана 3.12: у неё дольше история готовых wheel, и та же минорная версия
  стоит внутри образа Triton. Поэтому клиент и `model.py` работают на одном Python.
  Верхняя граница `<3.13` нужна, чтобы `uv` у проверяющего не взял другую версию.
- **Triton `26.11-py3`.** Последний релиз на момент начала работы. У тега есть сборки
  linux/amd64 и linux/arm64, поэтому на Mac образ запускается нативно, без эмуляции.
  Запасной вариант, если обнаружатся проблемы релиза: `26.10-py3`.
- **`tritonclient==2.75.0`** выбран под версию сервера в образе.

### Ресурсы

- Образ Triton: ~10.6 GB при скачивании, **28.1 GB** на диске.
- Docker Desktop: вход в аккаунт не нужен, образ `nvcr.io` скачивается анонимно.

### Платформа

- Проверено: `tritonserver` 26.11 запускается на macOS arm64 нативно
  (без `--platform linux/amd64`), Python backend работает. Предупреждение
  `NVIDIA Driver was not detected` ожидаемо: сервер работает на CPU.

### Известные риски / не проверено

- **`torch` и `faiss` не импортируются в одном процессе.** На macOS у каждого своя копия
  `libomp.dylib` (`torch/lib/` и `faiss/.dylibs/`). Если `faiss` загружен первым, свёртки
  `torch` падают с segfault (exit 139, воспроизведено). Поэтому эмбеддинги (`torch`)
  и FAISS-индекс считаются отдельными шагами, клиент использует только `torch`,
  а сервер Triton только `faiss`.
- В образе сервера стоит numpy 1.26.4, в клиенте 2.5.5. `docker/Dockerfile` закрепляет
  `faiss-cpu==1.16.0` и `numpy==1.26.4`, поэтому numpy в образе не обновляется.
  Индекс, записанный `faiss.write_index` на Mac (numpy 2.x), читается в контейнере
  без ошибок: формат файла FAISS от numpy не зависит.

## Данные

Stanford Online Products, официальный архив `Stanford_Online_Products.zip` по ссылке авторов
из [README репозитория статьи](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16#stanford-online-products-dataset).

- По умолчанию скачивается с Google Drive. FTP-ссылка `ftp://cs.stanford.edu/...` доступна
  как `--source ftp`, но в части сетей FTP блокируется (TCP-соединение есть,
  ответа сервера нет), поэтому она не используется по умолчанию.
- Размер 3 083 860 082 байта, sha256
  `b04fa78210e69fab050ce73939550e1cd0f96890bdd98ec9fee732c3b756ef48`. Скрипт проверяет оба значения.
  Авторы хеш не публикуют, он зафиксирован по первой загрузке. Независимая сверка:
  `Ebay_info.txt`, `Ebay_train.txt`, `Ebay_test.txt` побайтно совпадают (git blob SHA1)
  с копией `pawlo2013/StanfordOnlineProducts` на Hugging Face.
- Проверенные факты: 120 053 фото, 22 634 товара (`class_id`), 12 категорий (`super_class_id`).
- Распакованный датасет занимает ~2.9 GB в `data/raw/` (в git не попадает).

## Каталог / запросы

**ID объекта** = `image_id` из `Ebay_info.txt` (одно фото = один ID). ID задаёт сам датасет,
мы их не перенумеровываем. Товар (`class_id`) хранится в манифесте отдельно.

Правила (`src/sop_search/split.py`):
1. **Точные дубликаты** определяются по sha256 байтов файла. В группе одинаковых файлов
   остаётся фото с минимальным `image_id`, остальные получают роль `duplicate`
   и не попадают ни в каталог, ни в запросы. Так в индексе нет двух одинаковых файлов,
   и top-1 в self-retrieval однозначен.
2. **Запросы**: в каждой из 12 категорий `round(0.10 · N)` товаров отдают по одному фото,
   где N — число уникальных фото категории. Участвуют только товары с ≥2 уникальными фото,
   поэтому у каждого запроса в каталоге остаётся хотя бы одно фото того же товара.
3. **Детерминированно, без RNG**: товары сортируются по `sha256("sop-split-v1:class:<class_id>")`,
   фото внутри товара выбирается по минимальному `sha256("sop-split-v1:photo:<image_id>")`.
   При равенстве хешей решает целый ID. Встроенный `hash()` не используется, потому что
   он солится в каждом процессе.

Результат на полном датасете:

| Роль | Фото |
|---|---|
| catalog | **106 224** (все с уникальными байтами, ≥100 000) |
| query | 11 803 |
| duplicate | 2 026 (1 982 внутри одного товара; 44 — одно объявление в двух товарах разных категорий) |

- Манифест: `artifacts/manifest.csv`, sha256
  `ceae52a33b69e57848e578db432631b69bffd7952f0748668d41dc8f8de6461e`.
  Две сборки подряд дают побайтно одинаковый файл.
- Сборка занимает ~8 с на Apple Silicon (хеширование всех файлов).

## Прототип: выбор модели

Подмножество для прототипа (`prototype_subset` в `src/sop_search/split.py`): товары берутся
целиком в порядке `sha256("sop-split-v1:prototype:<class_id>")`, пока в каталоге не станет
≥1000 фото. Вышло 1003 фото каталога, 119 запросов, 215 товаров.

Эмбеддинг для обеих моделей L2-нормируется, мера сходства — косинус, то есть скалярное
произведение нормированных векторов (`IndexFlatIP`). Замер на CPU Apple Silicon,
8 потоков torch, батч 32 (`scripts/compare_models.py`, результаты в `artifacts/prototype/compare.json`):

| Модель | Ревизия HF | D | Параметры | мс/фото | self top-1 | Hit@5 | top-1 тот же товар |
|---|---|---|---|---|---|---|---|
| `openai/clip-vit-base-patch32` (image encoder + projection) | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | 512 | 87.8 M | **9.1** | 1.000 | **0.899** | **0.832** |
| `facebook/dinov2-small` (CLS после layernorm) | `ed25f3a31f01632728cabb09d1542f84ab7b0056` | 384 | 22.1 M | 15.3 | 1.000 | 0.815 | 0.714 |

- DINOv2-small медленнее при вчетверо меньшем числе параметров: патч 14 на кадре 224
  даёт 256 токенов, а у CLIP с патчем 32 их 49.
- Запросов всего 119, разница по Hit@5 (0.08) примерно два стандартных отклонения.
- Оценка для полного прогона (~118k фото): CLIP ≈ 18 мин, DINOv2-small ≈ 30 мин.
  Индекс FP32: 106 224 × 512 × 4 Б ≈ 218 MB (CLIP) или ≈ 163 MB (DINOv2-small).

**Выбрана CLIP ViT-B/32** (`DEFAULT_MODEL` в `src/sop_search/embedding.py`): на прототипе
она лучше и по Hit@5, и по скорости.

## Сервис

```
клиент: фото -> CLIP (общая предобработка) -> FP32[512], L2-норма
       -> Triton, модель knn_search (Python backend, FAISS загружается один раз в initialize())
       <- NEIGHBOR_IDS INT64[5], ближайший первым
```

- Индекс: `faiss.IndexIDMap2(faiss.IndexFlatIP(512))`, точный поиск. В индексе хранятся
  сами `image_id` (`add_with_ids`), поэтому сервер возвращает постоянные ID,
  а не номера строк FAISS.
- Вход `EMBEDDING` FP32 `[512]`, выход `NEIGHBOR_IDS` INT64 `[5]`, `max_batch_size: 0`.
- **Проверка размерности** делается в `config.pbtxt` (`dims: [ 512 ]`): Triton отклоняет
  вектор другой длины до вызова `model.py`. `model.py` дополнительно проверяет форму
  и NaN/Inf, а при старте сверяет размерность индекса с конфигом.
- Путь к индексу задаётся параметром `index_path` в `config.pbtxt` (`/index/index.faiss`).
  Какой индекс подключить, прототип или полный, определяет монтирование `-v ...:/index`.
- Сборка разделена на два процесса: `embed_catalog.py` (только `torch`) пишет
  `embeddings.npy` и `ids.npy`, затем `build_index.py` (только `faiss`) пишет `index.faiss`.

### Прототип end-to-end (1003 фото)

| Шаг | Результат |
|---|---|
| Эмбеддинги 1003 фото | 9.6 с (~9.6 мс/фото, 8 потоков) |
| Сборка индекса | <0.2 с, 2.1 MB; self top-1 на 1000 векторах: OK |
| Старт Triton до ready | ~4 с, лог: `loaded /index/index.faiss: 1003 vectors, dim 512` |
| Перезапуск контейнера | индекс читается из файла, ничего не пересчитывается |
| Запрос из каталога `29187` | top-1 = `29187` (он сам) |
| Запрос из пула запросов `29188` | `[29187, 29190, 86851, 109390, 30636]`: два первых — тот же товар `4753` |
| Вектор неверной длины `[3]` | ошибка `unexpected shape for input 'EMBEDDING' ... Expected [512], got [3]` |
| Следующий валидный запрос | 200, верный ответ |
| Задержка запроса по HTTP | медиана ~5.5 мс |
| RAM контейнера | ~86 MiB |

Пример: запрос `29188` (`chair_final/291308531568_1.JPG`, белый пластиковый стул снизу) →
первый ID `29187` (`chair_final/291308531568_0.JPG`, тот же стул спереди).

Оценка полного прогона (106 224 фото): эмбеддинги ≈ 17 мин, индекс ≈ 218 MB.

## Команды

```bash
# окружение
uv sync

# данные: скачать (~3 GB), проверить размер и sha256, распаковать в data/raw/
uv run python scripts/download_data.py

# манифест каталог/запросы -> artifacts/manifest.csv
uv run python scripts/build_manifest.py

# сравнение моделей на прототипе (1003 фото каталога + 119 запросов)
uv run python scripts/compare_models.py

# эмбеддинги (torch) и индекс (faiss): отдельные процессы
uv run python scripts/embed_catalog.py --subset prototype
uv run python scripts/build_index.py --dir artifacts/prototype

# сервер: образ Triton 26.11 + faiss-cpu
docker build -t sop-triton docker/
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/model_repository:/models:ro" \
  -v "$PWD/artifacts/prototype:/index:ro" \
  sop-triton tritonserver --model-repository=/models

# клиент: по ID из манифеста или по файлу
uv run python client/search.py --image-id 29188
uv run python client/search.py --image path/to/photo.jpg

# тесты
uv run pytest -q
```
