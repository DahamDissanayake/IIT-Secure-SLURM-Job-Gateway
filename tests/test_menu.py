# tests/test_menu.py
"""Main Menu wiring: item numbering and the Notify When Free gate."""
import inspect

import iitgpu.menu as m


def test_notify_when_free_is_a_main_menu_item():
    assert any("Notify When Free" in item for item in m._MAIN_ITEMS)
    assert any(item.startswith("5.") for item in m._MAIN_ITEMS)


def test_admin_item_renumbered_to_six_alongside_notify():
    src = inspect.getsource(m.run_menu)
    assert '"6. Admin' in src
    assert 'choice.startswith("6.")' in src
    assert 'choice.startswith("5.")' in src


def test_notify_menu_only_offers_subscribing_when_gpu_fully_occupied():
    """The subscribe flow must be gated on free == 0 -- if pods are already
    free there's nothing to wait for."""
    src = inspect.getsource(m._notify_menu)
    assert "if free > 0:" in src
    assert "Nothing to notify" in src


def test_notify_menu_uses_pod_notify_module():
    src = inspect.getsource(m._notify_menu)
    assert "pod_notify.subscribe" in src
    assert "pod_notify.unsubscribe" in src
    assert "pod_notify.get_subscription" in src


def test_notify_menu_requires_a_registered_email_before_subscribing():
    src = inspect.getsource(m._notify_menu)
    assert "daemonclient.email_for" in src
    assert "No registered email" in src
