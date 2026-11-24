"""Architecture boundaries and behavior preservation for the module move."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_active_code_has_no_legacy_or_processing_to_ui_imports():
    classes = {}
    for path in (ROOT / 'src').rglob('*.py'):
        if 'legacy' in path.parts or 'tests' in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or '']
            assert not any('legacy' in module.split('.') for module in modules), path
            if 'processing' in path.parts:
                assert not any(module.startswith('src.ui') for module in modules), path
            if 'segmentation' in path.parts:
                assert not any('datasets' in module.split('.') for module in modules), path
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                assert node.name not in classes, (node.name, classes.get(node.name), path)
                classes[node.name] = path
    assert not (ROOT / 'src/ui/pipeline.py').exists()
    assert not (ROOT / 'src/ui/config.py').exists()


def test_candidate_extraction_preserves_previous_records():
    from src.processing.segmentation.extraction import BoundaryCandidateExtractor
    from src.processing.segmentation.candidates import BoundaryCandidateDetector
    import hashlib

    # Compile only the original extraction methods; archive modules never enter imports.
    path = ROOT / 'src/legacy/previous_layout/processing/datasets/boundary_dataset_builder.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.body = [node for node in cls.body if isinstance(node, ast.Assign) or
                isinstance(node, ast.FunctionDef) and node.name in
                {'__init__', '_display_context', 'extract'}]
    namespace = {'hashlib': hashlib, 'BoundaryCandidateDetector': BoundaryCandidateDetector}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    previous = namespace['BoundaryDatasetBuilder']
    for language, text in [('ru', '— Привет! — сказал он.\nОна ответила: «Да».') ,
                           ('en', '“Hello!” he said.\nShe replied, “Yes.”'),
                           ('ru', ''), ('en', 'One.\tTwo.\u00a0Three!')]:
        assert BoundaryCandidateExtractor(12).extract(text, language, 'book') == previous(12).extract(text, language, 'book')


def test_archive_cannot_be_imported():
    import importlib
    with pytest.raises(ImportError, match='archive'):
        importlib.import_module('src.legacy')


def test_cli_dispatches_to_shared_runner():
    tree=ast.parse((ROOT/'src/processing/pipeline/cli.py').read_text())
    assert any(isinstance(n,ast.ImportFrom) and n.module=='runner' for n in ast.walk(tree))
    assert not any(isinstance(n,ast.FunctionDef) and n.name in ('process_document','align_documents') for n in ast.walk(tree))
