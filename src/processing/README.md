# Организация обработки текста

```text
processing/
  main.py                         стабильная точка входа CLI
  pipeline/
    cli.py                        прежний сценарий CLI
    config.py                     пути, документы и названия этапов
    runner.py                     диспетчер этапов для UI и фонового процесса
    io.py                         общие файловые операции
  preprocessing/
    stages.py                     запуск очистки и нормализации
    cleaning.py                   очистка исходного текста
    normalization.py              нормализация предложений
    export.py                     экспорт JSON → TXT для Vecalign
  segmentation/
    segmenter.py                  границы предложений, spaCy
    candidates.py                 обнаружение кандидатов границ
    extraction.py                 контексты и записи кандидатов для обоих языков
    stages.py                     запуск этапа сегментации
    english_dialogue.py           английские кавычки и слова автора
    ru_classifier.py              RuBERT и ограничения границ
    boundary_aware.py             объединение правил, модели и сегментатора
    postprocessor.py              восстановление пунктуации
  evaluation/
    windows.py                    окна из соседних групп Vecalign
    cometkiwi.py                   адаптер предзагруженной модели
    engine.py                     оценка подготовленных окон
    io.py                         JSON и CLI для произвольных пар
    stages.py                     этап COMETKiwi для main и UI
    README.md                     установка, запуск и формат результата
  datasets/
    boundary_dataset_builder.py   подготовка и проверка датасета границ
    data/training/                исходный учебный CSV
  alignment/
    stages.py                     запуск LASER, Vecalign, SimAlign и корпуса
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

`BoundaryCandidateExtractor` отвечает за обнаружение границ и контексты;
`BoundaryDatasetBuilder` наследует его и добавляет только чтение/запись учебного CSV.
Рабочая сегментация не зависит от модуля подготовки датасета.

UI импортирует конфигурацию и диспетчер из `src.processing.pipeline`; обратных
зависимостей от UI нет. Прежние `src/ui/config.py` и `src/ui/pipeline.py` удалены,
перенаправляющих копий классов нет. Сценарии CLI и UI сохраняют свои прежние
параметры и порядок операций; этот перенос не меняет алгоритмы сегментации.

Исторические исходники и старые самостоятельные проверки находятся в
`src/legacy` и не участвуют в импортах, запуске приложения или сборе pytest.
Актуальные проверки: `python -m pytest`; браузерная проверка включается
через `RUN_BROWSER_TESTS=1`.

Оценка перевода COMETKiwi: [инструкции](evaluation/README.md). Этап `cometkiwi`
доступен отдельно и включён в полный пайплайн; необходимы дополнительные зависимости
и checkpoint модели.
