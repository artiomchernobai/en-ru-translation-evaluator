# Выравнивание слов

Модуль принимает нормализованные документы и результат выравнивания предложений.
Для каждой непустой пары групп SimAlign строит связи между словами. Код ядра
не содержит имён произведений, переводчиков и файлов и не записывает данные.

## Установка и модель

В Python-окружении, **в котором запускаются main.py или UI worker**, установите
подходящий CPU/CUDA-вариант PyTorch, затем:

```powershell
python -m pip install -r src/processing/requirements-word-alignment.txt
```

Это отдельная зависимость: установленный в WSL LASER не устанавливает SimAlign
в Windows-окружение интерфейса. По умолчанию используется multilingual BERT
(`bert-base-multilingual-cased`), слой 8, CPU, три метода SimAlign: `inter`,
`mwmf`, `itermax`. Выбранные связи для токенов берутся из `itermax`.
Первый запуск скачивает модель Hugging Face; последующие используют кэш.

Необязательные переменные окружения:

- `SIMALIGN_MODEL`: `bert`, `xlmr`, идентификатор Hugging Face или путь к локальной
  модели с токенизатором;
- `SIMALIGN_DEVICE`: `cpu` (по умолчанию), `cuda` или устройство PyTorch;
- `SIMALIGN_METHOD`: `itermax`, `inter`, `mwmf`;
- `SIMALIGN_LAYER`: номер слоя модели (по умолчанию `8`).

Модель загружается один раз для всех пар в одном запуске этапа. Отдельный worker
при следующем запуске загружает модель снова из кэша.

## Запуск

В UI доступен этап **«SimAlign — выравнивание слов»**. Выберите оригинал и один
или несколько переводов. Требуются `data/normalized/*.json` и соответствующие
`data/aligned/*.align`. Результаты находятся в:

```text
data/word_alignment/original-rus_glenny-translation-en.json
data/word_alignment/original-rus_pevear-translation-en.json
```

Этап включён в полный запуск после Vecalign, перед сборкой корпуса. Его можно
запустить отдельно, без RuBERT, spaCy, LASER и повторного выравнивания предложений:

```powershell
python -m src.processing.main --stage word_align
# Или напрямую из корня проекта:
python src/processing/main.py --stage word_align
```

Для **любого другого произведения/перевода**, без изменения конфигурации проекта:

```powershell
python -m src.processing.word_alignment_io --source data/normalized/book-ru.json --target data/normalized/book-en.json --alignment data/aligned/book.align --output data/word_alignment/book-ru_book-en.json
```

Файлы JSON имеют существующий формат нормализатора:

```json
{
  "document": "book-ru",
  "language": "ru",
  "sentence_count": 1,
  "sentences": [{"sentence_id": 42, "normalized_text": "Привет, мир!"}]
}
```

Индексы `.align` — позиции в массиве `sentences`, а не значения `sentence_id`.
Исходные тексты не меняются. Файл результата заменяется атомарно после успешного
завершения: ошибка модели или входных данных сохраняет предыдущий результат.

## API без файлов

```python
from src.processing.word_alignment import WordAligner
from src.processing.simalign_backend import SimAlignBackend

aligner = WordAligner(SimAlignBackend(model="bert", device="cpu"))
result = aligner.align(
    source_document, target_document,
    [{"source_indices": [0], "target_indices": [0, 1], "cost": 0.25}],
)
# Выбор имён, путей и запись result реализуются вызывающим приложением.
```

`align_word_files` в `word_alignment_io.py` — отдельный файловый адаптер. Он
принимает все пути явно и может переиспользовать переданный объект `WordAligner`.
Для тестов/другой модели можно передать backend с `metadata` и методом
`align_tokens(source_tokens, target_tokens)`, возвращающим пары индексов слов.

## Структура результата, schema_version 1.0

- `source`, `target`: идентификатор и язык документа, отпечаток входных данных,
  нормализованный текст, предложения и массив токенов.
- Каждый токен: `token_id`, текст, `sentence_index`, исходный `sentence_id`,
  `index_in_sentence`, символьные границы в предложении и документе,
  `is_punctuation`, `aligned_token_ids`, `alignment_status`.
  Ссылки в `aligned_token_ids` указывают на токены **другой стороны**.
- `blocks`: `block_id` совпадает с `segment_id` корпуса, собранного из того же
  `.align`. Словесные связи сохраняются в отдельном файле, без дублирования
  массивов токенов в корпусе. Это группы Vecalign, включая группы с пустой одной стороной. Стороны
  хранят индексы/ID предложений, объединённый текст, `token_ids`, `token_spans`
  с границами токенов внутри объединённого текста. Связь 1:N не разрушает
  принадлежность токенов к отдельным предложениям.
- `word_alignment.methods`: связи каждого метода. В каждой связи есть локальные
  индексы внутри блока и глобальные `source_token_id`/`target_token_id`.
- `source_unlinked_token_ids`, `target_unlinked_token_ids`: слова без связей
  **выбранного метода**, только для действительно обработанных блоков.
- `backend`, `selected_method`: модель, слой, устройство, версия библиотеки,
  известная ревизия модели и выбранный метод. SimAlign не предоставляет
  вероятности правильности связей; вымышленные confidence не добавляются.
- `inputs`: пути и SHA-256 файлов (только файловый адаптер).
- `summary`: число блоков, обработанных групп, пустых сторон, слишком длинных
  блоков и выбранных словесных связей.

Все индексы начинаются с **0**. Смещения измеряются в Unicode code points,
правая граница исключена. Смещения документа относятся к нормализованным
предложениям, соединённым `\n`, блока — к предложениям, соединённым пробелом.
Они не относятся к исходному тексту до очистки; JavaScript использует UTF-16,
поэтому для символов вне BMP потребуется преобразование смещений.
Токенизатор выделяет Unicode-слова/числа, сохраняет внутренние апострофы и дефисы,
отделяет пунктуацию. Это слова, не внутренние BPE-токены модели; пробелы
восстанавливаются по тексту и смещениям.

## Состояния и ограничения

- `computed`: модель обработала весь блок; токены имеют состояние `linked`
  или `unlinked`.
- `not_applicable`: одна сторона выравнивания предложений пуста. Модель не
  вызывается; токены имеют состояние `sentence_unaligned`.
- `skipped_too_long`: блок не помещается в токеновый лимит модели. Весь текст и
  токены сохранены со статусом `not_processed`, ссылок нет. Никакого скрытого
  обрезания не происходит. Общий `processing_status` в этом случае — `partial`.

Несвязанный токен не является доказательством пропуска/добавления перевода.
Связи наследуют качество входных пар предложений, поэтому каждая группа имеет
`sentence_alignment_review: unreviewed`. `vecalign_cost` сохраняется отдельно
и не интерпретируется как уверенность. Ошибочная пара предложений не
исправляется автоматически на этапе выравнивания слов.
