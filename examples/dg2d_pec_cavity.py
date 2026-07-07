"""
Illustrative 2D DGTD simulation: PEC resonant cavity (Hesthaven & Warburton, p. 205).

Produces Ez field snapshots and correlation vs the analytical cavity mode.
Run from the repository root:

    python examples/dg2d_pec_cavity.py --output examples/output

This example is not part of the 1D local-stability analysis in the paper.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from maxwell.dg.dg2d import Maxwell2D
from maxwell.dg.mesh2d import readFromGambitFile
from maxwell.driver import MaxwellDriver

TEST_DATA = os.path.join(os.path.dirname(__file__), '..', 'testData')


def resonant_cavity_ez_field(x, y, t):
    """(1,1) TEz cavity mode on the mesh bounding box [xmin,xmax]x[ymin,ymax]."""
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    Lx = xmax - xmin
    Ly = ymax - ymin
    m, n = 1, 1
    xhat = (x - xmin) / Lx
    yhat = (y - ymin) / Ly
    omega = np.pi * np.sqrt((m / Lx) ** 2 + (n / Ly) ** 2)
    return (
        np.sin(m * np.pi * xhat)
        * np.sin(n * np.pi * yhat)
        * np.cos(omega * t)
    )


def run(
    mesh_name='Maxwell2D_K146.neu',
    n_order=5,
    n_steps=40,
    flux_type='Centered',
    cfl=1.0,
    output_dir=None,
):
    mesh_path = os.path.join(TEST_DATA, mesh_name)
    msh = readFromGambitFile(mesh_path)
    sp = Maxwell2D(n_order, msh, fluxType=flux_type)
    driver = MaxwellDriver(sp, CFL=cfl)

    driver['Ez'][:] = resonant_cavity_ez_field(sp.x, sp.y, 0.0)

    correlations = []
    times = []
    for _ in range(n_steps):
        t = driver.timeIntegrator.time
        ez_ref = resonant_cavity_ez_field(sp.x, sp.y, t)
        correlations.append(np.corrcoef(ez_ref.ravel(), driver['Ez'].ravel())[0, 1])
        times.append(t)
        driver.step()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    pcm = sp.plot_field(n_order, driver['Ez'], ax=axes[0])
    axes[0].triplot(sp.mesh.getTriangulation(), 'k-', lw=0.3, alpha=0.4)
    axes[0].set_title(f'Ez at t = {driver.timeIntegrator.time:.3f}')
    fig.colorbar(pcm, ax=axes[0], fraction=0.046)

    axes[1].plot(times, correlations, 'b.-')
    axes[1].set_xlabel('time')
    axes[1].set_ylabel('correlation vs analytical mode')
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True)
    axes[1].set_title('Mode correlation over time')

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, 'dg2d_pec_cavity.png')
        plt.savefig(out, dpi=150)
        print(f'Saved {out}')
    else:
        plt.show()

    print(f'Final correlation: {correlations[-1]:.4f}')
    return correlations[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D PEC cavity DGTD demonstration')
    parser.add_argument('--mesh', default='Maxwell2D_K146.neu', choices=[
        'Maxwell2D_K8.neu', 'Maxwell2D_K146.neu'
    ])
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--flux', default='Centered', choices=['Centered', 'Upwind'])
    parser.add_argument('--cfl', type=float, default=1.0)
    parser.add_argument('--output', default=None, help='Directory to save figure')
    args = parser.parse_args()
    run(
        mesh_name=args.mesh,
        n_steps=args.steps,
        flux_type=args.flux,
        cfl=args.cfl,
        output_dir=args.output,
    )
