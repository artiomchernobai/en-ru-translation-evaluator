"""Opt-in real browser regression: RUN_BROWSER_TESTS=1 pytest tests/browser -q."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import pytest
playwright = pytest.importorskip('playwright.sync_api')
pytestmark = pytest.mark.skipif(os.environ.get('RUN_BROWSER_TESTS') != '1', reason='Opt-in Chromium/Streamlit test')


@pytest.fixture
def application(tmp_path):
    from src.processing.alignment.words import WordAligner
    class Backend:
        metadata = {'name': 'browser-test'}
        def align_tokens(self, source, target):
            return {'itermax': [(0, 0), (1, 1)], 'inter': [(0, 0)], 'mwmf': [(0, 0), (1, 1)]}
    def doc(name, lang, texts):
        return {'document': name, 'language': lang, 'sentences': [
            {'sentence_id': i, 'normalized_text': text} for i, text in enumerate(texts)]}
    src = doc('test-source', 'ru', ['Он читает книгу.', 'Он читает книгу.', '😀 Мир! <script>alert(1)</script>'])
    tgt = doc('test-target', 'en', ['He reads books.', 'He reads books.', '😀 World! <script>alert(1)</script>'])
    data = WordAligner(Backend()).align(src, tgt, [
        {'source_indices': [i], 'target_indices': [i], 'cost': .2} for i in range(3)])
    output = tmp_path/'data/word_alignment/test.json'
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    from src.processing.evaluation.windows import build_windows
    from src.processing.evaluation.engine import evaluate_windows
    class QualityBackend:
        metadata = {'name': 'browser-QE-test'}
        def predict(self, pairs):
            return [{'status': 'scored', 'score': .75} for _ in pairs]
    quality = evaluate_windows(build_windows(src, tgt, [
        {'source_indices': [i], 'target_indices': [i], 'cost': .2} for i in range(3)]), QualityBackend())
    quality.update(source_document=src['document'], target_document=tgt['document'], radius=1)
    from src.processing.analysis.candidates import collect_candidates
    candidates = collect_candidates(data, quality)
    candidate_output = tmp_path/'data/review_candidates/test.candidates.json'
    candidate_output.parent.mkdir(parents=True)
    candidate_output.write_text(json.dumps(candidates), encoding='utf-8')
    quality_output = tmp_path/'data/translation_quality/test.cometkiwi.json'
    quality_output.parent.mkdir(parents=True)
    quality_output.write_text(json.dumps(quality), encoding='utf-8')
    for document in (src, tgt):
        path = tmp_path/f"data/raw/{document['document']}.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_text('\n\n'.join(s['normalized_text'] for s in document['sentences']), encoding='utf-8')
    jobs=tmp_path/'data/ui_jobs';jobs.mkdir()
    (jobs/'example.status.json').write_text(json.dumps({'state':'done'}))
    (jobs/'example.log').write_text('Test worker completed successfully.\n'*20)
    repo = Path(__file__).resolve().parents[2]
    script = tmp_path/'application.py'
    script.write_text(f'import sys\nsys.path.insert(0,{str(repo)!r})\nfrom src.ui.app import main\nfrom pathlib import Path\nmain(Path({str(tmp_path)!r}))\n')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    with (tmp_path/'server.log').open('w') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(script),
            '--server.address', '127.0.0.1', '--server.port', str(port), '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'], stdout=log, stderr=log, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        try:
            deadline = time.monotonic()+30
            while time.monotonic()<deadline:
                try:
                    with urllib.request.urlopen(url+'/_stcore/health', timeout=1) as response:
                        if response.status==200: break
                except OSError:
                    if proc.poll() is not None: pytest.fail((tmp_path/'server.log').read_text())
                    time.sleep(.1)
            else: pytest.fail('Streamlit did not become ready')
            yield url
        finally:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill();proc.wait()


def test_click_links_themes_layout_and_safe_text(application):
    with playwright.sync_playwright() as p:
        executable = os.environ.get('CHROMIUM_PATH') or shutil.which('chromium')
        with p.chromium.launch(executable_path=executable, headless=True, args=['--no-sandbox']) as browser:
            page = browser.new_page(viewport={'width': 1600, 'height': 1000})
            errors, dialogs = [], []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.on('dialog', lambda d: (dialogs.append(d.message), d.dismiss()))
            page.goto(application)
            page.get_by_text('Проверка', exact=True).click()
            page.locator('.review').wait_for()
            page.wait_for_function("parseFloat(getComputedStyle(document.querySelector('h1')).fontSize) < 24")
            title = page.get_by_role('heading', name='Text Pipeline · Workbench').bounding_box()
            header = page.get_by_test_id('stHeader').bounding_box()
            assert title['y'] >= header['y']+header['height']
            assert not page.get_by_text('Окно A', exact=True).count()
            source, target = page.locator('section.pane').nth(0), page.locator('section.pane').nth(1)
            source.locator('[data-id="1"]').click()
            playwright.expect(target.locator('[data-id="1"]')).to_contain_class('related')
            source.locator('[data-id="2"]').click()
            playwright.expect(page.get_by_role('status').last).to_contain_text('Связей не найдено')
            target.locator('[data-id="5"]').click()
            playwright.expect(source.locator('[data-id="5"]')).to_contain_class('related')
            assert source.locator('.text').inner_text().count('\n\n')==2
            assert '<script>alert(1)</script>' in source.locator('.text').inner_text()
            for theme, color in [('Dark', 'rgb(255, 255, 255)'), ('Light', 'rgb(0, 0, 0)')]:
                if not page.get_by_test_id('stMainMenuItem-theme-'+theme).is_visible():
                    page.get_by_role('button', name='Main menu').click()
                page.get_by_test_id('stMainMenuItem-theme-'+theme).click()
                playwright.expect(source.locator('[data-id="5"]')).to_contain_class('related')
                page.wait_for_function("document.documentElement.style.getPropertyValue('--reading-color') === "+repr('#ffffff' if theme=='Dark' else '#000000'))
                assert source.locator('.text').evaluate('(e)=>getComputedStyle(e).color')==color
                assert source.locator('[data-id="5"]').evaluate('(e)=>getComputedStyle(e).color')==color
            page.keyboard.press('Escape')
            before = page.locator('.review').bounding_box()['height']
            page.get_by_role('button', name='▴ Журнал выполнения', exact=True).click()
            page.set_viewport_size({'width': 1280, 'height': 720})
            playwright.expect(page.locator('.review')).to_be_visible()
            page.wait_for_timeout(250)
            assert page.locator('.review').bounding_box()['height'] < before
            assert page.locator('.review').bounding_box()['y']+page.locator('.review').bounding_box()['height'] <= page.get_by_test_id('stBottom').bounding_box()['y']
            page.get_by_role('combobox').nth(1).click()
            page.get_by_role('option', name='inter', exact=True).click()
            playwright.expect(source.locator('[data-id="1"]')).to_contain_class('unlinked')
            source.locator('[data-id="0"]').focus()
            page.keyboard.press('Enter')
            playwright.expect(target.locator('[data-id="0"]')).to_contain_class('related')
            assert not dialogs and not errors
            assert not page.get_by_test_id('stException').count()


def test_quality_window_navigation_and_scope(application):
    with playwright.sync_playwright() as p:
        executable = os.environ.get('CHROMIUM_PATH') or shutil.which('chromium')
        with p.chromium.launch(executable_path=executable, headless=True, args=['--no-sandbox']) as browser:
            page = browser.new_page(viewport={'width': 1600, 'height': 1000})
            page.goto(application)
            page.get_by_text('Оценка перевода', exact=True).click()
            playwright.expect(page.get_by_text('Оценка всего окна · выше — лучше', exact=True)).to_be_visible()
            playwright.expect(page.get_by_test_id('stMetricValue')).to_have_text('0.7500')
            playwright.expect(page.get_by_text('Все записи Vecalign, использованные для оценки окна', exact=True)).to_be_visible()
            playwright.expect(page.get_by_role('button', name='COMETKiwi — оценка перевода', exact=True)).to_have_count(1)
            assert not page.get_by_test_id('stException').count()


def test_candidate_evidence_highlighting_and_safe_text(application):
    with playwright.sync_playwright() as p:
        executable = os.environ.get('CHROMIUM_PATH') or shutil.which('chromium')
        with p.chromium.launch(executable_path=executable, headless=True, args=['--no-sandbox']) as browser:
            page = browser.new_page(viewport={'width': 1600, 'height': 1000})
            dialogs = []
            page.on('dialog', lambda d: (dialogs.append(d.message), d.dismiss()))
            page.goto(application)
            page.get_by_text('Кандидаты', exact=True).click()
            playwright.expect(page.get_by_role('heading', name='Кандидаты на проверку', exact=True)).to_be_visible()
            playwright.expect(page.locator('mark').first).to_be_visible()
            playwright.expect(page.get_by_text('Записи Vecalign и контекст окна', exact=True)).to_be_visible()
            playwright.expect(page.get_by_role('button', name='Кандидаты на проверку', exact=True)).to_have_count(1)
            assert not page.get_by_test_id('stException').count()
            assert not dialogs
