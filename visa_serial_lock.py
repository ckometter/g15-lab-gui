"""Notice when NI-VISA's serial-port lock is stuck, so serial devices fail fast instead of hanging.

NI-VISA (21.5 on the lab PC) guards every serial (ASRL) search and open with one
mutex shared by all programs, ``Global\\NIVISA_SYSTEM_SEMAPHORES #2``. When the
program holding it gets stuck, every ``list_resources()`` that includes serial
ports and every ``open_resource('ASRL...')`` waits forever, in any program; GPIB,
USB and TCPIP are unaffected. On 17 Nov 2026 the holder was NI Configuration
Manager (nimxs.exe), the service behind NI MAX.

The check only reads the mutex state. It never takes the lock and never opens a
port. Where the mutex does not exist (no NI-VISA, another VISA, not Windows),
there is nothing to check and devices connect as before.
"""
import functools
import sys
import time

SERIAL_LOCK_NAME = "Global\\NIVISA_SYSTEM_SEMAPHORES #2"
WAIT_S = 5.0          # a normal holder releases it within milliseconds
RECHECK_S = 60.0      # after a stuck result, the next device fails at once

FIX = ("This happens when NI Configuration Manager (nimxs.exe) gets stuck. In PowerShell opened "
       "with 'Run as administrator', run  Stop-Process -Name nimxs -Force  and then  "
       "Start-Service mxssvr, nidevldu, NITaggerService  (or restart the PC). Then press Connect / Retry.")


class SerialLockHeld(RuntimeError):
    """NI-VISA's serial-port lock stayed taken, so opening a serial device would hang."""


@functools.lru_cache(maxsize=1)
def _windows_api():
    import ctypes
    from ctypes import wintypes

    class MutantBasicInformation(ctypes.Structure):
        _fields_ = [("current_count", wintypes.LONG), ("owned_by_caller", wintypes.BOOLEAN),
                    ("abandoned", wintypes.BOOLEAN)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    kernel32.OpenMutexW.restype = wintypes.HANDLE
    kernel32.OpenMutexW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    ntdll.NtQueryMutant.restype = wintypes.LONG
    ntdll.NtQueryMutant.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.ULONG, ctypes.c_void_p]
    return ctypes, kernel32, ntdll, MutantBasicInformation


def serial_lock_owned(name=SERIAL_LOCK_NAME):
    """True if the lock is taken right now, False if it is free, None if it cannot be checked."""
    if sys.platform != "win32":
        return None
    ctypes, kernel32, ntdll, MutantBasicInformation = _windows_api()
    handle = kernel32.OpenMutexW(0x00100000 | 0x0001, False, name)  # SYNCHRONIZE | MUTANT_QUERY_STATE
    if not handle:
        return None  # NI-VISA has not created it: nothing can be holding it
    try:
        info = MutantBasicInformation()
        if ntdll.NtQueryMutant(handle, 0, ctypes.byref(info), ctypes.sizeof(info), None) != 0:
            return None
        return info.current_count <= 0  # 1 = free; 0 or less = owned
    finally:
        kernel32.CloseHandle(handle)


_stuck_at = None


def wait_for_serial_lock(device, *, wait=WAIT_S, poll=0.1, owned=serial_lock_owned,
                         clock=time.monotonic, sleep=time.sleep):
    """Return once the lock is free or cannot be checked; raise SerialLockHeld if it stays taken.

    It waits up to ``wait`` seconds, except right after another device found the lock
    stuck: then it fails at once, so Check All Devices does not wait once per device.
    """
    global _stuck_at
    start = clock()
    state = owned()
    if state and _stuck_at is not None and start - _stuck_at < RECHECK_S:
        wait = 0.0
    while state:
        if clock() - start >= wait:
            _stuck_at = clock()
            raise SerialLockHeld(
                f"{device} was not opened: another program holds NI-VISA's serial-port lock and did "
                f"not release it, so opening a serial port would hang. GPIB and USB devices are "
                f"not affected. " + FIX)
        sleep(poll)
        state = owned()
    _stuck_at = None
    return state
