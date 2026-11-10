
"""Postprocess sentence records without throwing away punctuation."""

from __future__ import annotations

import json
import re
from pathlib import Path


class SentencePostprocessor:

    PUNCTUATION_ONLY = re.compile(
        r"""^[\s'"“”‘’«»„.!?…,:;()\[\]{}\-–—]+$"""
    )

    OPENING = frozenset("—–-«“‘„(")

    @staticmethod
    def _attach_opening(
        prefix: str,
        text: str
    ) -> str:
        """
        Присоединяет отдельно выделенную
        открывающую пунктуацию к следующему сегменту.
        """

        # После диалогового тире необходим пробел.
        # Кавычки присоединяются непосредственно к слову.

        if (
            prefix.endswith(("—", "–", "-"))
            and text
            and not text[0].isspace()
        ):
            return prefix + " " + text

        return prefix + text

    @staticmethod
    def _has_unclosed_quote(
        fragment: str,
        quote: str
    ) -> bool:
        """
        Проверяет наличие незакрытой прямой кавычки.

        Апострофы внутри сокращений не учитываются:
            don't
            I'm
            Beer's

        Используется как резервный механизм
        после сегментации.
        """

        count = 0

        for index, char in enumerate(fragment):

            if char != quote:
                continue

            if (
                quote == "'"
                and 0 < index < len(fragment) - 1
            ):

                before = fragment[index - 1]
                after = fragment[index + 1]

                is_contraction = (
                    before.isalnum()
                    and after.isalnum()
                    and not (
                        before.islower()
                        and after.isupper()
                    )
                )

                if is_contraction:
                    continue

            count += 1

        return count % 2 == 1

    @classmethod
    def _repair_leading_closing_quote(
        cls,
        previous: str,
        current: str
    ) -> tuple[str, str]:
        """
        Исправляет закрывающую кавычку,
        ошибочно перенесённую в следующий сегмент.

        Было:
            "'Hello!"
            "' The people left."

        Стало:
            "'Hello!'"
            "The people left."
        """

        if (
            not previous
            or not current
            or current[0] not in "'\""
        ):
            return previous, current

        if (
            len(current) < 2
            or not current[1].isspace()
        ):
            return previous, current

        # Предыдущее предложение должно заканчиваться
        # терминальной пунктуацией.

        if not re.search(
            r"[.!?…]$",
            previous
        ):
            return previous, current

        # Если предыдущий сегмент содержит
        # незакрытую кавычку, текущая кавычка,
        # вероятно, принадлежит ему.

        if not cls._has_unclosed_quote(
            previous,
            current[0]
        ):
            return previous, current

        previous = previous + current[0]

        current = current[1:].lstrip()

        return previous, current

    def process(
        self,
        sentences: list[dict]
    ) -> list[dict]:
        """
        Постобработка сегментов.

        1. Удаляет пустые сегменты.
        2. Присоединяет отдельно выделенную пунктуацию.
        3. Исправляет закрывающие английские кавычки.
        4. Восстанавливает дефисные слова и междометия.
        5. Пересчитывает sentence_id.
        """

        segments: list[str] = []

        pending_open = ""

        for item in sentences:

            text = str(
                item.get("text", "")
            ).strip()

            if not text:
                continue

            # 1. Сегмент состоит только из пунктуации.

            if self.PUNCTUATION_ONLY.fullmatch(text):

                if all(
                    char in self.OPENING
                    or char.isspace()
                    for char in text
                ):

                    pending_open += text.strip()

                    continue

                if pending_open:

                    pending_open += text

                    continue

                if segments:

                    segments[-1] += text

                else:

                    pending_open += text

                continue

            # 2. Присоединяем открывающую пунктуацию.

            if pending_open:

                text = self._attach_opening(
                    pending_open,
                    text
                )

                pending_open = ""

            # 3. Исправляем закрывающую кавычку,
            # ошибочно присоединённую к следующему
            # предложению.

            if segments:

                segments[-1], text = (
                    self._repair_leading_closing_quote(
                        segments[-1],
                        text
                    )
                )

            if not text:
                continue

            # 4. Восстанавливаем слова, разделённые
            # на границе дефиса.
            #
            # Пример:
            # "А-" + "а!" -> "А-а!"

            if (
                segments
                and re.search(
                    r"[A-Za-zА-Яа-яЁё]-$",
                    segments[-1]
                )
                and re.match(
                    r"^[a-zа-яё]",
                    text
                )
            ):

                segments[-1] += text

            # 5. Восстанавливаем междометия,
            # ошибочно разорванные ПЕРЕД дефисом.
            #
            # Пример:
            # "– А" + "- а!" -> "– А-а!"

            elif (
                segments
                and re.search(
                    r"(?:^|\s)[А-ЯЁA-Z]$",
                    segments[-1]
                )
                and re.match(
                    r"^-[ \t]*[а-яёa-z]",
                    text
                )
            ):

                repaired = re.sub(
                    r"^-[ \t]*",
                    "-",
                    text,
                    count=1
                )

                segments[-1] += repaired

            else:

                segments.append(text)

        # 6. Если осталась отдельная пунктуация,
        # сохраняем её, а не удаляем.

        if pending_open:

            if segments:

                segments[-1] += pending_open

            else:

                segments.append(pending_open)

        # 7. Обновляем идентификаторы предложений.

        return [
            {
                "sentence_id": i,
                "text": text
            }
            for i, text in enumerate(segments)
        ]

    def process_file(
        self,
        input_path,
        output_path=None
    ):
        """
        Обрабатывает JSON-файл с предложениями.

        Если output_path не указан,
        создаётся файл с суффиксом _postprocessed.json.
        """

        source_path = Path(input_path)

        target_path = (
            Path(output_path)
            if output_path
            else source_path.with_name(
                source_path.stem
                + "_postprocessed.json"
            )
        )

        with source_path.open(
            "r",
            encoding="utf-8-sig"
        ) as stream:

            data = json.load(stream)

        data["sentences"] = self.process(
            data["sentences"]
        )

        data["sentence_count"] = len(
            data["sentences"]
        )

        target_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with target_path.open(
            "w",
            encoding="utf-8"
        ) as stream:

            json.dump(
                data,
                stream,
                ensure_ascii=False,
                indent=4
            )

        return target_path
