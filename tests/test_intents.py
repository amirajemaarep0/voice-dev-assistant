"""Classifying what the developer asked for.

Misclassification is expensive: a "write me a test" question routed to the
generic path gets a chatty summary instead of code, and a style question
routed to the syntax path runs the wrong tool entirely.
"""
from __future__ import annotations

import pytest

from assistant import intents


class TestDetectIntent:
    @pytest.mark.parametrize("question", [
        "write a unit test for build_index",
        "generate a test for the function add",
        "create a test case for Calculator",
        "can you write me some pytest tests?",
        "unit test for average please",
    ])
    def test_test_requests(self, question):
        assert intents.detect_intent(question).kind == intents.TEST

    @pytest.mark.parametrize("question", [
        "is there a syntax error in app.py?",
        "what is wrong with samples/broken_function.py?",
        "what's wrong with this file",
        "does app.py compile?",
        "any errors in my project?",
        "is this broken?",
        "find the bug in indexer.py",
    ])
    def test_syntax_requests(self, question):
        assert intents.detect_intent(question).kind == intents.SYNTAX

    @pytest.mark.parametrize("question", [
        "how should messy_style.py be formatted?",
        "reformat this file",
        "does app.py follow PEP 8?",
        "run the linter on store.py",
        "check the indentation in stt.py",
        "is my code style consistent?",
    ])
    def test_style_requests(self, question):
        assert intents.detect_intent(question).kind == intents.STYLE

    @pytest.mark.parametrize("question", [
        "explain the function normalize_project_path",
        "what does build_index do?",
        "how does retrieval work?",
        "describe the Assistant class",
    ])
    def test_explain_requests(self, question):
        assert intents.detect_intent(question).kind == intents.EXPLAIN

    @pytest.mark.parametrize("question", ["", "   ", "app.py", "hello"])
    def test_falls_back_to_general(self, question):
        assert intents.detect_intent(question).kind == intents.GENERAL

    def test_test_beats_explain_when_both_words_appear(self):
        """"explain how to test X" is still a request to write a test."""
        assert intents.detect_intent(
            "explain and write a unit test for add"
        ).kind == intents.TEST


class TestExtractSymbol:
    @pytest.mark.parametrize("question,expected", [
        ("explain the function normalize_project_path", "normalize_project_path"),
        ("explain `iter_source_files`", "iter_source_files"),
        ('what does "build_index" do?', "build_index"),
        ("what does Calculator.accumulate do?", "Calculator.accumulate"),
        ("what does read_text() return?", "read_text"),
        ("write a unit test for average", "average"),
        ("describe the class ProjectStore", "ProjectStore"),
    ])
    def test_finds_the_identifier(self, question, expected):
        assert intents.extract_symbol(question) == expected

    @pytest.mark.parametrize("question", [
        "what is in app.py?",
        "check samples/broken_function.py",
        "explain the code",
        "how does it work?",
        "",
    ])
    def test_no_identifier_to_find(self, question):
        assert intents.extract_symbol(question) == ""

    def test_a_filename_is_never_a_symbol(self):
        """Filenames go down the file-reference path, not the symbol path."""
        assert intents.extract_symbol("what is wrong with broken_function.py") == ""

    def test_symbol_is_attached_to_the_intent(self):
        intent = intents.detect_intent("write a unit test for build_index")
        assert intent.symbol == "build_index"
        assert intent.needs_symbol

    def test_syntax_intent_carries_no_symbol(self):
        assert intents.detect_intent("syntax error in add_numbers?").symbol == ""
