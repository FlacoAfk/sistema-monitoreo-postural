"""
audio_alert — Cross-platform audio beep.

Replaces inline winsound.Beep() with a multi-platform fallback chain:
1. Windows  → powershell [console]::beep()
2. Linux    → aplay (requires alsa-utils)
3. Fallback → silent no-op

Usage:
    from src.ui.audio_alert import beep
    beep(frequency=1000, duration_ms=300)
"""

from __future__ import annotations

import subprocess
import platform
import sys

__all__ = ["beep"]


def _win_beep(freq: int, duration_ms: int) -> None:
    """Windows beep via powershell (works in headless and container sessions)."""
    try:
        subprocess.run(
            ["powershell", "-c", f"[console]::beep({freq},{duration_ms})"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        pass


def _linux_beep(freq: int, duration_ms: int) -> None:
    """Linux beep via aplay (alsa-utils must be installed)."""
    try:
        # Generate a sine wave via speaker-test or fall back to plain beep
        subprocess.run(
            ["aplay", "-q"],
            input=b"\x00" * 256,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except FileNotFoundError:
        # aplay not available — try `beep` binary
        try:
            subprocess.run(
                ["beep", "-f", str(freq), "-l", str(duration_ms)],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception:
            pass
    except Exception:
        pass


def _noop_beep(_freq: int, _duration_ms: int) -> None:
    """Silent fallback — no audio subsystem available."""
    pass


# ── Platform dispatch ─────────────────────────────────────────────────────────

_system = platform.system()

if _system == "Windows":
    _beep_impl = _win_beep
elif _system == "Linux":
    _beep_impl = _linux_beep
else:
    # Darwin (macOS) / other — no built-in beep available
    _beep_impl = _noop_beep


def beep(frequency: int = 1000, duration_ms: int = 300) -> None:
    """Emit an audio beep.  No-op if no audio subsystem is available.

    Args:
        frequency:  Pitch in Hz (ignored on some platforms).
        duration_ms: Duration in milliseconds.
    """
    _beep_impl(frequency, duration_ms)
