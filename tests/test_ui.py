from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iitgpu import ui


def test_style_is_one_shared_object():
    assert ui.STYLE is not None
    # Same object every import — not re-built per module.
    from iitgpu.ui import STYLE as style_again
    assert ui.STYLE is style_again


def test_back_constants_use_only_the_permitted_arrow_glyph():
    assert ui.BACK == "← Back"
    assert ui.BACK_TO_MAIN == "← Back to main menu"


def test_screen_renders_a_panel_with_the_title(monkeypatch):
    buf = Console(file=None, force_terminal=True, width=80)
    captured = []
    monkeypatch.setattr(ui, "console", type("C", (), {
        "print": lambda self, renderable: captured.append(renderable)
    })())
    ui.screen("My Screen")
    assert len(captured) == 1
    assert isinstance(captured[0], Panel)


def test_screen_accepts_a_status_body(monkeypatch):
    captured = []
    monkeypatch.setattr(ui, "console", type("C", (), {
        "print": lambda self, renderable: captured.append(renderable)
    })())
    table = Table()
    ui.screen("Admin", status=table)
    assert captured[0].renderable is table


def test_select_menu_returns_none_on_back(monkeypatch):
    class FakeSelect:
        def __init__(self, *a, **kw):
            self.kw = kw
        def ask(self):
            return ui.BACK
    monkeypatch.setattr(ui.questionary, "select", lambda *a, **kw: FakeSelect(*a, **kw))
    result = ui.select_menu("Pick:", ["a", "b"])
    assert result is None


def test_select_menu_returns_none_on_escape(monkeypatch):
    class FakeSelect:
        def ask(self):
            return None
    monkeypatch.setattr(ui.questionary, "select", lambda *a, **kw: FakeSelect())
    assert ui.select_menu("Pick:", ["a", "b"]) is None


def test_select_menu_returns_the_real_choice(monkeypatch):
    class FakeSelect:
        def ask(self):
            return "a"
    monkeypatch.setattr(ui.questionary, "select", lambda *a, **kw: FakeSelect())
    assert ui.select_menu("Pick:", ["a", "b"]) == "a"


def test_select_menu_appends_separator_and_back_to_choices(monkeypatch):
    seen = {}
    class FakeSelect:
        def ask(self):
            return None
    def fake_select(prompt, choices, style):
        seen["choices"] = choices
        seen["style"] = style
        return FakeSelect()
    monkeypatch.setattr(ui.questionary, "select", fake_select)
    ui.select_menu("Pick:", ["a", "b"])
    assert seen["choices"][-1] == ui.BACK
    assert seen["style"] is ui.STYLE
