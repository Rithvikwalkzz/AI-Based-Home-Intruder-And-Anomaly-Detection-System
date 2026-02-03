# continuous_beep.py
"""
Continuous Windows beep module.
Use:
    from continuous_beep import start_beeping, stop_beeping
    start_beeping()    # start alarm
    stop_beeping()     # stop alarm
"""

import sys
import time
import threading

IS_WINDOWS = sys.platform.startswith("win")

if IS_WINDOWS:
    import winsound

_beep_running = False
_beep_thread = None


def _beep_loop():
    """Internal: nonstop beep while _beep_running is True."""
    while _beep_running:
        try:
            winsound.Beep(1200, 200)  # main beep
            winsound.Beep(900, 200)   # secondary tone
            time.sleep(0.05)
        except:
            pass


def start_beeping():
    """Start continuous beeping (ignored if already running)."""
    global _beep_running, _beep_thread

    if not IS_WINDOWS:
        print("[Beep] Not a Windows system. Skipping.")
        return

    if _beep_running:
        return  # already active

    _beep_running = True
    _beep_thread = threading.Thread(target=_beep_loop, daemon=True)
    _beep_thread.start()
    print("[Beep] Continuous alarm started.")


def stop_beeping():
    """Stop the continuous beep."""
    global _beep_running

    if _beep_running:
        _beep_running = False
        print("[Beep] Alarm stopped.")
