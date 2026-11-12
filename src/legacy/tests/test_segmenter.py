
import pytest

from src.processing.segmentation.segmenter import SentenceSegmenter

TEST_CASES = [
    {
        "name": "dialogue_with_speech_verb",
        "text": "'No,' said Berlioz.",
        "expected": [
            "'No,' said Berlioz."
        ]
    },
    {
        "name": "dialogue_with_proper_name",
        "text": "– No! – Ivan replied.",
        "expected": [
            "– No! – Ivan replied."
        ]
    },
    {
        "name": "dialogue_with_pronoun",
        "text": "– No! – He replied.",
        "expected": [
            "– No!",
            "– He replied."
        ]
    },
    {
        "name": "two_dialogue_sentences",
        "text": "'Why?' asked Ivan. 'Because it is impossible.'",
        "expected": [
            "'Why?' asked Ivan.",
            "'Because it is impossible.'"
        ]
    },
    {
        "name": "dialogue_with_proper_name",
        "text": "– No! – Ivan replied.",
        "expected": ["– No! – Ivan replied."]
    },
    {
        "name": "dialogue_with_pronoun",
        "text": "– No! – He replied.",
        "expected": ["– No!", "– He replied."]
    },
    {
        "name": "dialogue_with_lowercase",
        "text": "– No! – the stranger replied.",
        "expected": ["– No! – the stranger replied."]
    }
]

@pytest.fixture(scope="module")
def segmenter():
    return SentenceSegmenter()


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "– Реплика! – сказал Иван.",
            ["– Реплика! – сказал Иван."]
        ),
        (
            "– Реплика! – Иван ответил.",
            ["– Реплика! – Иван ответил."]
        ),
        (
            "– Реплика! – Он ответил.",
            ["– Реплика!", "– Он ответил."]
        ),
        (
            "– Реплика! – Ответ.",
            ["– Реплика!", "– Ответ."]
        ),
        (
            "– Реплика! – Ответ. – сказал кто-то.",
            ["– Реплика!", "– Ответ. – сказал кто-то."]
        ),
        (
            "«Куда ты идешь?!» — грозно спросил отец.",
            ["«Куда ты идешь?!» — грозно спросил отец."]
        )
    ]
)
def test_russian_dialogue_boundaries(
    segmenter,
    text,
    expected
):

    result = segmenter.segment(text, "ru")

    actual = [
        sentence["text"]
        for sentence in result
    ]

    assert actual == expected

@pytest.mark.parametrize("case", TEST_CASES)
def test_sentence_segmentation(segmenter, case):
    result = segmenter.segment(case["text"], "en")

    actual = [item["text"] for item in result]

    assert actual == case["expected"], (
        f"\nTest: {case['name']}"
        f"\nExpected: {case['expected']}"
        f"\nActual: {actual}"
    )

def test_debug_dialogue_boundary(segmenter):
    nlp = segmenter.models["en"]

    with nlp.select_pipes(disable=["dialogue_boundary_resolver"]):
        doc = nlp("– No! – Ivan replied.")

        for token in doc:
            print(
                repr(token.text),
                "POS:", token.pos_,
                "ENT:", token.ent_type_,
                "START:", token.is_sent_start
            )

import tempfile
from pathlib import Path
from src.processing.datasets.boundary_dataset_builder import BoundaryDatasetBuilder

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