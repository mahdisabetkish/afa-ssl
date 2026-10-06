"""Thread caps, memory limits and machine-health checks.

Import this module (and call `configure_threads`) before numpy or torch are
imported anywhere in a process; the BLAS thread pools read the environment
variables at import time.
"""
from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS",
)


def configure_threads(n: int = 4) -> None:
    """Export every thread cap so BLAS, OpenMP and torch see the same limit."""
    for var in THREAD_VARS:
        os.environ[var] = str(n)
    try:
        import torch  # noqa: WPS433 (deliberate late import)

        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def limit_memory(max_bytes: int = 6 * 1024**3) -> None:
    """Cap the process heap. RLIMIT_DATA covers brk and private anonymous
    mappings on Linux >= 4.7, which is where numpy and torch allocate, and it
    does not interfere with the large virtual reservations that RLIMIT_AS
    would break."""
    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    new_soft = max_bytes if soft == resource.RLIM_INFINITY else min(soft, max_bytes)
    resource.setrlimit(resource.RLIMIT_DATA, (new_soft, hard))


@dataclass
class MachineState:
    load1: float
    temp_c: float | None
    rss_gb: float
    avail_gb: float


def read_max_temp() -> float | None:
    base = "/sys/class/thermal"
    temps: list[float] = []
    try:
        for name in os.listdir(base):
            if not name.startswith("thermal_zone"):
                continue
            with open(os.path.join(base, name, "temp")) as fh:
                temps.append(int(fh.read().strip()) / 1000.0)
    except OSError:
        return None
    return max(temps) if temps else None


def machine_state() -> MachineState:
    import psutil

    proc = psutil.Process()
    return MachineState(
        load1=os.getloadavg()[0],
        temp_c=read_max_temp(),
        rss_gb=proc.memory_info().rss / 1024**3,
        avail_gb=psutil.virtual_memory().available / 1024**3,
    )


def wait_until_cool(
    max_load: float = 6.0,
    max_temp: float = 85.0,
    min_avail_gb: float = 1.0,
    poll_s: float = 15.0,
    max_wait_s: float = 1800.0,
    log=print,
) -> None:
    """Block while the machine is hot, overloaded or short on memory."""
    waited = 0.0
    while waited < max_wait_s:
        st = machine_state()
        hot = st.temp_c is not None and st.temp_c > max_temp
        busy = st.load1 > max_load
        tight = st.avail_gb < min_avail_gb
        if not (hot or busy or tight):
            return
        log(
            f"machine busy (load1={st.load1:.2f}, temp={st.temp_c}, "
            f"avail={st.avail_gb:.2f} GB); sleeping {poll_s:.0f}s"
        )
        time.sleep(poll_s)
        waited += poll_s
