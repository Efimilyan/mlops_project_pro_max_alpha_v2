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

Машина разработки: macOS, Apple Silicon (arm64), без GPU.

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

- На macOS arm64 резолвер выбирает более старые `torch` / `faiss-cpu`, чем на Linux
  (2.11.0 / 1.12.0 против 2.15.0 / 1.16.0). Нужно решить, фиксировать ли одинаковые версии.
- В образе numpy 1.26.4, локально резолвится numpy 2.x. Совместимость Python backend
  с numpy 2.x после установки `faiss-cpu` в образ нужно проверить.

## Команды

```bash
# окружение
uv sync

# образ сервера
docker pull nvcr.io/nvidia/tritonserver:26.11-py3
```
