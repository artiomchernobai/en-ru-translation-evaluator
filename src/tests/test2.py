import tempfile
from pathlib import Path
from src.processing.boundry_dataset_builder import BoundaryDatasetBuilder

builder = BoundaryDatasetBuilder(context_size=120)

def fragments(text, language):
    result = builder.extract(text, language, 'test')
    return [(text[:row['position']].rstrip(), text[row['position']:].lstrip()) for row in result]

# Both of these must be found as candidates. A model/human labels one 1, other 0.
assert ('– Дайте нарзану,', '– попросил Берлиоз.') in fragments('– Дайте нарзану, – попросил Берлиоз.', 'ru')
assert ('...за Садовое кольцо,', '– никто не пришел.') in fragments('...за Садовое кольцо, – никто не пришел.', 'ru')

long_ru = '– Но, позвольте вас спросить, – после тревожного раздумья спросил заграничный гость, – как же быть?'
assert len(fragments(long_ru, 'ru')) == 2
assert fragments('«Немец» – подумал Берлиоз.', 'ru') == [('«Немец»', '– подумал Берлиоз.')]
assert fragments('Он ответил: – Нет.', 'ru') == [('Он ответил:', '– Нет.')]

sample = "The avenue was empty. 'A glass of lemonade, please,'said Berlioz."
assert fragments(sample, 'en') == [
    ('The avenue was empty.', "'A glass of lemonade, please,'said Berlioz."),
    ("The avenue was empty. 'A glass of lemonade, please,'", 'said Berlioz.')
]
assert len(fragments("'Hello. ', said John.", 'en')) == 1
assert len(fragments("'A hundred per cent! ' affirmed the poet.", 'en')) == 1
assert fragments("He said: `Hello,' she replied.", 'en') == [
    ('He said:', "`Hello,' she replied."),
    ("He said: `Hello,'", 'she replied.')
]
assert len(fragments("Don't worry, it's fine.", 'en')) == 0

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root/'ru.txt').write_text(long_ru, encoding='utf-8')
    (root/'en.txt').write_text(sample, encoding='utf-8')
    path = root/'candidates.csv'
    rows = builder.build({
        'original': {'path': root/'ru.txt', 'language': 'ru'},
        'glenny': {'path': root/'en.txt', 'language': 'en'}
    }, path)
    assert builder.validate_csv(path, rows) == len(rows)
print('All tests passed')