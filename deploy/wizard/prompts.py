"""Zero-dependency interactive prompt helpers.

The wizard must run on a completely fresh machine before slurm-deck's own
dependencies (rich, questionary) are installed -- so this uses only stdlib
input()/print(), with plain ANSI color codes matching the ok()/warn()/fail()/
step() convention already used across deploy/*.sh.
"""
import getpass
import sys

_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _c(code: str, text_: str) -> str:
    return f"{code}{text_}{_RESET}" if _supports_color() else text_


def step(msg: str) -> None:
    print(f"\n{_c(_BOLD + _CYAN, '==>')} {msg}")


def ok(msg: str) -> None:
    print(f"  {_c(_GREEN, '✔')}  {msg}")


def warn(msg: str) -> None:
    print(f"  {_c(_YELLOW, '⚠')}  {msg}")


def fail(msg: str) -> None:
    print(f"  {_c(_RED, '✘')}  {msg}", file=sys.stderr)


def text(prompt: str, default: str | None = None, validator=None,
         allow_empty: bool = False) -> str:
    """Prompt for a text value. `validator(value)` returns an error string
    to reject the value and re-prompt, or None/"" to accept it. If
    allow_empty, a blank answer (with no default) is accepted as-is instead
    of being re-prompted -- for genuinely optional fields."""
    suffix = f" [{default}]" if default not in (None, "") else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        val = raw or (default or "")
        if not val and not allow_empty:
            print("  a value is required.")
            continue
        if val and validator:
            err = validator(val)
            if err:
                print(f"  {err}")
                continue
        return val


def yesno(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        val = input(f"{prompt}{suffix}: ").strip().lower()
        if not val:
            return default
        if val in ("y", "yes"):
            return True
        if val in ("n", "no"):
            return False
        print("  please answer y or n.")


def choice(prompt: str, options: list[str], default: int = 0) -> int:
    print(prompt)
    for i, opt in enumerate(options, 1):
        marker = "  (default)" if i - 1 == default else ""
        print(f"  {i}. {opt}{marker}")
    while True:
        val = input(f"Choose [1-{len(options)}]: ").strip()
        if not val:
            return default
        if val.isdigit() and 1 <= int(val) <= len(options):
            return int(val) - 1
        print("  invalid choice.")


def secret(prompt: str) -> str:
    while True:
        val = getpass.getpass(f"{prompt}: ").strip()
        if val:
            return val
        print("  a value is required.")


def confirm_risky(description: str) -> bool:
    """Checkpoint printed before any root/sudo/system-mutating action."""
    print()
    print(_c(_BOLD + _YELLOW, "  ABOUT TO: ") + description)
    return yesno("  Proceed?", default=True)
