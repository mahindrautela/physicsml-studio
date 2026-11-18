"""Create two small demo datasets so the app works before anything is downloaded.

  kolmogorov2d.npy  2D forced Navier-Stokes (Kolmogorov flow), velocity u and v,
                    shape (10, 21, 2, 128, 128) - same layout as MORPH's FNS-KF demo
  burgers1d.npy     1D viscous Burgers equation, shape (16, 21, 256)

Both are simulated here with a pseudo-spectral solver (about half a minute on a
laptop). Usage:  python tools/make_demo_data.py [--out workspace/demo] [--force]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def kolmogorov(n_traj=10, n=128, steps_out=21, dt=0.005, out_every=0.5, spinup=5.0,
               nu=1e-3, drag=0.1, kf=4, amp=0.2, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n, 1.0 / n)
    kr = np.fft.rfftfreq(n, 1.0 / n)
    KY, KX = np.meshgrid(k, kr, indexing="ij")  # (n, n//2+1)
    K2 = KX ** 2 + KY ** 2
    K2_inv = np.where(K2 == 0, 0.0, 1.0 / np.where(K2 == 0, 1, K2))
    dealias = (np.abs(KX) < n / 3) & (np.abs(KY) < n / 3)
    y = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # body force amp*sin(kf*y) in x; its curl drives the vorticity
    force = np.fft.rfft2(np.broadcast_to((-amp * kf * np.cos(kf * y))[:, None], (n, n)))

    # random smooth initial vorticity, different per trajectory
    w_hat = (rng.standard_normal((n_traj, n, n // 2 + 1)) + 1j * rng.standard_normal((n_traj, n, n // 2 + 1)))
    w_hat *= np.exp(-K2 / 16.0)
    w_hat[:, 0, 0] = 0
    w0 = np.fft.irfft2(w_hat, s=(n, n))
    w_hat = np.fft.rfft2(w0 / w0.std(axis=(1, 2), keepdims=True))  # vorticity std 1

    lin = nu * K2 + drag

    def nonlinear(wh):
        psi = wh * K2_inv
        u = np.fft.irfft2(1j * KY * psi, s=(n, n))
        v = np.fft.irfft2(-1j * KX * psi, s=(n, n))
        wx = np.fft.irfft2(1j * KX * wh, s=(n, n))
        wy = np.fft.irfft2(1j * KY * wh, s=(n, n))
        return np.fft.rfft2(-(u * wx + v * wy)) * dealias + force

    def velocity(wh):
        psi = wh * K2_inv
        return np.stack([np.fft.irfft2(1j * KY * psi, s=(n, n)), np.fft.irfft2(-1j * KX * psi, s=(n, n))], axis=1)

    n_spin = int(round(spinup / dt))
    n_out = int(round(out_every / dt))
    out = np.empty((n_traj, steps_out, 2, n, n), dtype=np.float32)
    prev = nonlinear(w_hat)
    step = 0
    total = n_spin + n_out * (steps_out - 1)
    for s in range(total + 1):
        if s >= n_spin and (s - n_spin) % n_out == 0:
            out[:, (s - n_spin) // n_out] = velocity(w_hat)
            step += 1
        if s == total:
            break
        nl = nonlinear(w_hat)
        # Crank-Nicolson on viscosity/drag, Adams-Bashforth 2 on advection + forcing
        w_hat = ((1 - 0.5 * dt * lin) * w_hat + dt * (1.5 * nl - 0.5 * prev)) / (1 + 0.5 * dt * lin)
        prev = nl
    return out


def burgers(n_traj=16, n=256, steps_out=21, dt=2e-4, out_every=0.1, nu=0.02, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = np.fft.rfftfreq(n, 1.0 / n)
    dealias = np.abs(k) < n / 3
    u0 = np.zeros((n_traj, n))
    for m in range(1, 5):
        a = rng.normal(0, 1.0 / m, (n_traj, 1))
        ph = rng.uniform(0, 2 * np.pi, (n_traj, 1))
        u0 += a * np.sin(m * x[None] + ph)
    u_hat = np.fft.rfft(u0)
    lin = nu * k ** 2

    def nl(uh):
        u = np.fft.irfft(uh, n=n)
        return -0.5j * k * np.fft.rfft(u * u) * dealias

    n_out = int(round(out_every / dt))
    out = np.empty((n_traj, steps_out, n), dtype=np.float32)
    prev = nl(u_hat)
    for s in range(n_out * (steps_out - 1) + 1):
        if s % n_out == 0:
            out[:, s // n_out] = np.fft.irfft(u_hat, n=n)
        cur = nl(u_hat)
        u_hat = ((1 - 0.5 * dt * lin) * u_hat + dt * (1.5 * cur - 0.5 * prev)) / (1 + 0.5 * dt * lin)
        prev = cur
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "workspace" / "demo"))
    ap.add_argument("--force", action="store_true", help="Rebuild even if the files exist")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {"kolmogorov2d.npy": kolmogorov, "burgers1d.npy": burgers}
    for name, fn in jobs.items():
        f = out / name
        if f.exists() and not a.force:
            print(f"✓ {f} already exists")
            continue
        t0 = time.time()
        print(f"… simulating {name}", flush=True)
        arr = fn()
        if not np.all(np.isfinite(arr)):
            raise SystemExit(f"{name}: simulation blew up (non-finite values)")
        np.save(f, arr)
        print(f"✓ {f}  shape {arr.shape}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
