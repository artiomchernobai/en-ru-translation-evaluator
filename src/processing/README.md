# Организация обработки текста

```text
processing/
  main.py                         CLI и запуск этапов для выбранных документов
  preprocessing/
    cleaning.py                   очистка исходного текста
    normalization.py              нормализация предложений
    export.py                     экспорт JSON → TXT для Vecalign
  segmentation/
    segmenter.py                  границы предложений, spaCy
    candidates.py                 кандидаты границ
    english_dialogue.py           английские кавычки и слова автора
    ru_classifier.py              RuBERT и ограничения границ
    boundary_aware.py             объединение правил, модели и сегментатора
    postprocessor.py              восстановление пунктуации
  datasets/
    boundary_dataset_builder.py   подготовка и проверка датасета границ
    data/training/                исходный учебный CSV
  alignment/
    sentences.py                  единый парсер и проверки .align
    embeddings.py                 LASER (отдельное окружение)
    words.py                      независимое ядро словесного выравнивания
    simalign_backend.py           адаптер SimAlign
    io.py                         файлы и CLI для произвольных пар
    corpus.py                     сборка корпуса
    requirements.txt              зависимости SimAlign
    README.md                     формат словесного JSON и команды
```

Публичная точка запуска проекта сохранена:

```powershell
python -m src.processing.main
python src/processing/main.py --stage word_align
```

Для любых имён произведений и переводов:

```powershell
python -m src.processing.alignment.io --source data/normalized/book-ru.json --target data/normalized/book-en.json --alignment data/aligned/book.align --output data/word_alignment/book.json
```

Импорты теперь указывают на тематические подпакеты, например:

```python
from src.processing.preprocessing.cleaning import clean_text
from src.processing.segmentation.segmenter import SentenceSegmenter
from src.processing.datasets.boundary_dataset_builder import BoundaryDatasetBuilder
from src.processing.alignment.words import WordAligner
from src.processing.alignment.io import align_word_files
```

Старые пути модулей из корня `processing` заменены этими путями; код проекта,
тесты и команды внешних инструментов обновлены. Файл
`processing/requirements-word-alignment.txt` оставлен как совместимая ссылка на
новые зависимости. LASER запускается через `-m src.processing.alignment.embeddings`
из корня проекта, как напрямую, так и через WSL. Символьные индексы и формат
словесного JSON не менялись; ранее созданные результаты можно открывать без
повторного выравнивания.
