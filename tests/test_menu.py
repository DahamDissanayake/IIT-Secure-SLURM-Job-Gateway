# tests/test_menu.py
"""Main Menu wiring: item numbering and the Notify When Free gate."""
import inspect

import slurmdeck.menu as m


def test_notify_when_free_is_a_main_menu_item():
    assert any("Notify When Free" in item for item in m._MAIN_ITEMS)
    assert any(item.startswith("5.") for item in m._MAIN_ITEMS)


def test_admin_item_renumbered_to_six_alongside_notify():
    src = inspect.getsource(m.run_menu)
    assert '"6. Admin' in src
    assert 'choice.startswith("6.")' in src
    assert 'choice.startswith("5.")' in src


def test_notify_menu_only_skips_subscribing_when_every_pod_is_free():
    """Someone with 1-2 pods free must still be able to subscribe for a
    bigger threshold (e.g. wanting 3-4) -- only skip the subscribe flow
    when there's truly nothing left to wait for (free == total)."""
    src = inspect.getsource(m._notify_menu)
    assert "if free >= total:" in src
    assert "if free > 0:" not in src
    assert "Nothing to wait for" in src


def test_notify_menu_uses_pod_notify_module():
    src = inspect.getsource(m._notify_menu)
    assert "pod_notify.subscribe" in src
    assert "pod_notify.unsubscribe" in src
    assert "pod_notify.get_subscription" in src


def test_notify_menu_annotates_pod_choices_with_live_availability():
    src = inspect.getsource(m._notify_menu)
    assert "available now" in src
    assert "waiting" in src
    assert "k <= free" in src


def test_notify_menu_requires_a_registered_email_before_subscribing():
    src = inspect.getsource(m._notify_menu)
    assert "daemonclient.email_for" in src
    assert "No registered email" in src
