#!/usr/bin/env python
"""GPU preflight: verify EVERY visible CUDA device can actually create a context.

Shared a100_tandon nodes are EXCLUSIVE_PROCESS; SLURM occasionally hands us a
GPU that already has another job's context on it. `torch.cuda.set_device(i)` on
such a GPU dies with CUDA error 46 (cudaErrorDevicesUnavailable) the instant
training starts, wasting the whole allocation. This probe fails fast (exit 1) so
the sbatch can `scontrol requeue` for a fresh, clean allocation instead.

Exit 0 = all visible GPUs usable. Exit 1 = at least one is busy/unavailable.
Prints per-GPU compute mode + a one-line verdict for the log.
"""
import subprocess
import sys


def compute_modes():
    """uuid -> compute_mode, best-effort (empty on any failure)."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=uuid,compute_mode,memory.used',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        rows = {}
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                rows[parts[0]] = (parts[1], parts[2])
        return rows
    except Exception:
        return {}


def main():
    import torch
    if not torch.cuda.is_available():
        print('[preflight] FAIL: torch.cuda.is_available() == False', flush=True)
        return 1

    n = torch.cuda.device_count()
    modes = compute_modes()
    print(f'[preflight] visible CUDA devices: {n}', flush=True)

    bad = []
    for i in range(n):
        uuid = None
        try:
            uuid = torch.cuda.get_device_properties(i).uuid
            uuid = f'GPU-{uuid}' if uuid and not str(uuid).startswith('GPU-') else str(uuid)
        except Exception:
            uuid = '<unknown>'
        mode, memused = modes.get(uuid, ('?', '?'))
        try:
            torch.cuda.set_device(i)
            x = torch.zeros(8, device=f'cuda:{i}')
            x = x + 1.0
            torch.cuda.synchronize(i)
            del x
            print(f'[preflight]   cuda:{i} OK   mode={mode} mem_used={memused} uuid={uuid}',
                  flush=True)
        except Exception as e:
            bad.append(i)
            print(f'[preflight]   cuda:{i} BUSY mode={mode} mem_used={memused} uuid={uuid} '
                  f'-> {type(e).__name__}: {e}', flush=True)

    if bad:
        print(f'[preflight] FAIL: {len(bad)}/{n} GPU(s) unusable: {bad}', flush=True)
        return 1
    print(f'[preflight] PASS: all {n} GPU(s) usable', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
