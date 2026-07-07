"""
Illustrative 2D FDTD simulation: TE-mode Gaussian pulse on a PEC square cavity.

Run from the repository root:

    python examples/fd2d_te_pulse.py [--output DIR]

This example is not part of the 1D local-stability analysis in the paper.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from maxwell.driver import MaxwellDriver
from maxwell.fd.fd2d import FD2D


def run(final_time=2.0, kx_elem=80, output_dir=None):
    sp = FD2D(x_min=-1.0, x_max=1.0, kx_elem=kx_elem, boundary_labels='PEC')
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    xH, yH = np.meshgrid(sp.xH, sp.yH)
    s0 = 0.25
    initial_h = np.exp(-(xH ** 2 + yH ** 2) / (2 * s0 ** 2))
    driver['H'][:, :] = initial_h

    n_frames = 5
    frame_times = np.linspace(0, final_time, n_frames + 1)[1:]
    frames = []
    idx = 0
    while driver.timeIntegrator.time < final_time:
        driver.step()
        if idx < len(frame_times) and driver.timeIntegrator.time >= frame_times[idx]:
            frames.append((driver.timeIntegrator.time, driver['H'].copy()))
            idx += 1

    if not frames:
        frames.append((driver.timeIntegrator.time, driver['H'].copy()))

    fig, axes = plt.subplots(1, len(frames), figsize=(3 * len(frames), 3))
    if len(frames) == 1:
        axes = [axes]
    for ax, (t, h) in zip(axes, frames):
        im = ax.contourf(xH, yH, h, levels=50, cmap='viridis')
        ax.set_title(f'H at t={t:.2f}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle('2D TE FDTD — Gaussian pulse (PEC walls)')
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, 'fd2d_te_pulse.png')
        plt.savefig(out, dpi=150)
        print(f'Saved {out}')
    else:
        plt.show()

    r = np.corrcoef(initial_h.ravel(), driver['H'].ravel())[0, 1]
    print(f'Final correlation with initial pulse: {r:.4f}')
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D TE FDTD pulse demonstration')
    parser.add_argument('--time', type=float, default=2.0)
    parser.add_argument('--cells', type=int, default=80)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    run(final_time=args.time, kx_elem=args.cells, output_dir=args.output)
