# %% [markdown]
# # DGTD reflection of a conductor backed plasma slab (d = 10 cm)
#
# Comparison of the reflection of a bounded plasma slab covering a perfect
# conductor, computed in the time domain with the DGTD solver of PyDG1D,
# against the analytical result of
#
# > C. X. Yuan, Z. X. Zhou and H. G. Sun, "Reflection Properties of
# > Electromagnetic Wave in a Bounded Plasma Slab", IEEE Transactions on
# > Plasma Science, vol. 38, no. 12, pp. 3348-3355, 2010,
# > doi: 10.1109/TPS.2010.2084110
#
# The analytical reference is the cold, collisional plasma permittivity of
# Eq. (10) of the paper,
#
# $$ \varepsilon = 1 - \frac{\omega_p^2}{\omega^2 + \nu_{en}^2}
#                   - i \frac{\nu_{en} \omega_p^2}{\omega (\omega^2 + \nu_{en}^2)} $$
#
# and the reflection coefficient of a slab of thickness $d$ over a conductor,
# Eq. (11),
#
# $$ r = \frac{\tanh(i 2 \pi f d \sqrt{\varepsilon} / c) - \sqrt{\varepsilon}}
#             {\tanh(i 2 \pi f d \sqrt{\varepsilon} / c) + \sqrt{\varepsilon}} $$
#
# with the reflected power Eq. (12), $R = |r|^2$.
#
# The time domain solver integrates the single pole Lorentz model
#
# $$ \partial_t P = J, \qquad
#    \partial_t J = -2 \gamma J - \omega_1^2 P + \omega_p^2 E, \qquad
#    D = \varepsilon_\infty E + P $$
#
# (Jiang et al., "Dispersion analysis of finite difference and discontinuous
# Galerkin schemes for Maxwell's equations in linear Lorentz media", 2018).
# The cold plasma of Yuan2010 is recovered with $\omega_1 = 0$,
# $\varepsilon_\infty = 1$ and $\gamma = \nu_{en}/2$.  PyDG1D uses units where
# $c = 1$ and lengths are metres, so angular frequencies are divided by $c$.
#
# This file uses the "percent" cell format: run it in one go with
# `python examples/plot_yuan2010_dgtd_10cm.py`, or cell by cell in Jupyter or
# VS Code.  It writes `doc/fig/yuan2010_dgtd_10cm.png` and an animated GIF of
# the illumination and reflection `doc/fig/yuan2010_dgtd_10cm.gif`.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## Repository bootstrap
#
# The `maxwell` package lives in the repository root, which is not installed;
# in a notebook there is no `__file__`, so the root is searched upwards from
# the current directory.

# %%
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:  # notebook: no __file__
    PROJECT_ROOT = Path.cwd()
    while not (PROJECT_ROOT / "maxwell").is_dir():
        PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maxwell.driver import MaxwellDriver
from maxwell.dg.dg1d import DG1D
from maxwell.dg.mesh1d import Mesh1D

from skrf.frequency import Frequency
from skrf.media import Media

print(f"repository root: {PROJECT_ROOT}")

# %% [markdown]
# ## Plasma and slab parameters
#
# Argon plasma of Fig. 3 of Yuan2010: `f_p = 2 GHz`, `nu_en = 5 GHz`, slab
# thickness `d = 10 cm`, reflection band 2-4 GHz.

# %%
SPEED_OF_LIGHT = 299792458.0        # m/s
VACUUM_IMPEDANCE = 376.730313668    # ohm

PLASMA_FREQUENCY = 2.0e9            # Hz, f_p = omega_p / (2 pi)
COLLISION_FREQUENCY = 5.0e9         # s^-1, nu_en
SLAB_THICKNESS = 0.10               # m
FREQUENCIES = np.linspace(2.0e9, 4.0e9, 201)

# Normalized (c = 1, lengths in metres) Lorentz model parameters.
OMEGA_P = 2.0 * np.pi * PLASMA_FREQUENCY / SPEED_OF_LIGHT
GAMMA = 0.5 * COLLISION_FREQUENCY / SPEED_OF_LIGHT

# %% [markdown]
# ## Analytical reference (Yuan2010)
#
# The Drude permittivity of Eq. (10) is plugged into a scikit-rf transmission
# line of thickness `d`, terminated by a short circuit.  Renormalizing to the
# impedance of free space gives Eq. (11), and the reflected power Eq. (12).

# %%
class PlasmaMedia(Media):
    """Cold plasma as a scikit-rf media (Eq. (10) of Yuan2010)."""

    def __init__(self, frequency, plasma_frequency, collision_frequency,
                 **kwargs):
        super().__init__(frequency=frequency, **kwargs)
        self.plasma_frequency = plasma_frequency
        self.collision_frequency = collision_frequency

    @property
    def ep_r(self):
        omega = 2.0 * np.pi * self.frequency.f
        omega_p = 2.0 * np.pi * self.plasma_frequency
        return 1.0 - omega_p**2 / (omega * (omega - 1j * self.collision_frequency))

    @property
    def gamma(self):
        k0 = 2.0 * np.pi * self.frequency.f / SPEED_OF_LIGHT
        return 1j * k0 * np.sqrt(self.ep_r)

    @property
    def z0_characteristic(self):
        return VACUUM_IMPEDANCE / np.sqrt(self.ep_r)


def analytical_reflection_db(frequencies):
    """Reflected power [dB] of the conductor backed plasma slab, Eqs. (10-12)."""
    frequency = Frequency.from_f(np.asarray(frequencies, dtype=float),
                                 unit='hz')
    media = PlasmaMedia(frequency=frequency,
                        plasma_frequency=PLASMA_FREQUENCY,
                        collision_frequency=COLLISION_FREQUENCY)
    slab = media.line(d=SLAB_THICKNESS, unit='m') ** media.short()
    slab.renormalize(VACUUM_IMPEDANCE)
    return 10.0 * np.log10(np.abs(slab.s[:, 0, 0])**2)


analytical_db = analytical_reflection_db(FREQUENCIES)
print(f"analytical dip: {analytical_db.min():.2f} dB "
      f"@ {FREQUENCIES[np.argmin(analytical_db)] / 1e9:.3f} GHz")

# %% [markdown]
# ## DGTD simulation
#
# A right going Gaussian pulse is launched in vacuum and hits the plasma slab
# (the last 20 elements of the mesh), which is terminated by a PEC.  A second
# simulation of the same pulse in vacuum provides the incident field at the
# probe, so the reflected field is `E_total - E_reflected_free`.

# %%
def dgtd_simulation(plasma, n_order=2, tmax=2.5, n_frames=0):
    """Run the DG scheme with the Lorentz RHS.

    Returns ``(time, E at the probe, dt, sp, snapshots, snapshot times)``.
    When ``n_frames`` is positive, snapshots of the electric field are
    recorded for the movie.
    """
    if plasma:
        xmax, elements = 1.0, 200
        boundary = {"LEFT": "ABC", "RIGHT": "PEC"}
    else:
        xmax, elements = 3.0, 600
        boundary = {"LEFT": "ABC", "RIGHT": "ABC"}

    lorentz = None
    if plasma:
        elements_in_slab = int(round(SLAB_THICKNESS / (xmax / elements)))
        omega_p = np.zeros(elements)
        gamma = np.zeros(elements)
        omega_p[-elements_in_slab:] = OMEGA_P
        gamma[-elements_in_slab:] = GAMMA
        lorentz = {"omega_p": omega_p, "gamma": gamma}

    sp = DG1D(n_order, Mesh1D(0.0, xmax, elements, boundary_label=boundary),
              lorentz=lorentz)
    driver = MaxwellDriver(sp, timeIntegratorType='LSERK4', CFL=1.0)

    source_x, source_width = 0.25, 0.02
    driver['E'][:] = np.exp(-(sp.x - source_x)**2 / (2 * source_width**2))
    driver['H'][:] = driver['E']

    # Probe at x = 0.6 m (element 120 at the coarsest mesh), in vacuum.
    element, node = int(round(0.6 / (xmax / elements))), 0

    steps = int(tmax / driver.dt)
    stride = max(1, steps // n_frames) if n_frames else 0
    time = np.zeros(steps)
    signal = np.zeros(steps)
    snapshots = []
    snapshot_times = []
    for n in range(steps):
        driver.step()
        time[n] = (n + 1) * driver.dt
        signal[n] = driver['E'][node, element]
        if stride and n % stride == 0:
            snapshots.append(driver['E'].copy())
            snapshot_times.append(time[n])

    return time, signal, driver.dt, sp, snapshots, snapshot_times


def dgtd_probe(plasma, **kwargs):
    """Convenience wrapper returning only (time, signal, dt)."""
    time, signal, dt, _, _, _ = dgtd_simulation(plasma, **kwargs)
    return time, signal, dt


time, reflected_free, dt = dgtd_probe(plasma=False)
_, total, _ = dgtd_probe(plasma=True)

# %% [markdown]
# ## Reflection coefficient from the probe signal
#
# The DFT of the reflected field divided by the DFT of the incident field,
# evaluated at the physical frequencies (the phase uses `t / c` because the
# solver works in `c = 1` light-metre units).

# %%
def reflection_from_probe(time, signal, reference, dt, frequencies=FREQUENCIES):
    """Reflected power [dB] from the incident and total probe signals."""
    def dft(values, frequency):
        return np.sum(values * np.exp(
            -2j * np.pi * (frequency / SPEED_OF_LIGHT) * time) * dt)

    incident = np.array([dft(reference, f) for f in frequencies])
    reflected = np.array([dft(signal - reference, f) for f in frequencies])
    return 20.0 * np.log10(np.abs(reflected) / np.abs(incident))


numerical_db = reflection_from_probe(time, total, reflected_free, dt)

# %% [markdown]
# ## Comparison

# %%
deviation = numerical_db - analytical_db
print(f"DGTD vs Yuan2010: max |error| = {np.max(np.abs(deviation)):.3f} dB, "
      f"RMS = {np.sqrt(np.mean(deviation**2)):.3f} dB")
print(f"DGTD dip: {numerical_db.min():.2f} dB "
      f"@ {FREQUENCIES[np.argmin(numerical_db)] / 1e9:.3f} GHz")

# %%
figure, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(FREQUENCIES / 1e9, analytical_db, color="black", linewidth=2.0,
        label="Yuan2010 analytical, Eqs. (10-12)")
ax.plot(FREQUENCIES / 1e9, numerical_db, color="tab:red", linewidth=1.4,
        linestyle="--", label="DGTD numerical (PyDG1D)")

ax.set_xlim(2.0, 4.0)
ax.set_ylim(-30.0, 0.0)
ax.set_xlabel("Frequency / GHz")
ax.set_ylabel("Total Reflection [dB]")
ax.set_title("Reflection of a 10 cm plasma slab over a conductor")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", framealpha=0.95)
figure.tight_layout()

output = PROJECT_ROOT / "doc" / "fig" / "yuan2010_dgtd_10cm.png"
output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output, dpi=150)
print(f"saved {output.relative_to(PROJECT_ROOT)}")
plt.show()

# %% [markdown]
# ## Movie: illumination and reflection
#
# Snapshots of the electric field during the same simulation.  The Gaussian
# pulse travels to the right (illumination), enters the plasma slab (shaded)
# and the dispersively reflected pulse travels back.

# %%
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

_, _, _, sp, snapshots, snapshot_times = dgtd_simulation(
    plasma=True, tmax=1.6, n_frames=240)

x_nodes = sp.x.ravel(order='F')
field_frames = [frame.ravel(order='F') for frame in snapshots]

figure_movie, ax = plt.subplots(figsize=(9, 4))
ax.axvspan(0.9, 1.0, color="tab:orange", alpha=0.25, label="plasma slab")
ax.axvline(1.0, color="black", linestyle=":", linewidth=1.0, label="PEC")
ax.axvline(0.6, color="tab:gray", linestyle="--", linewidth=0.8,
           label="reflection probe")
line, = ax.plot([], [], color="tab:blue", linewidth=1.2)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x / m")
ax.set_ylabel("E")
ax.set_title("DGTD electric field, 10 cm plasma slab over a conductor")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
time_text = ax.text(0.5, 0.96, "", transform=ax.transAxes,
                    ha="center", va="top")
figure_movie.tight_layout()


def animate(i):
    line.set_data(x_nodes, field_frames[i])
    time_text.set_text(f"t = {snapshot_times[i]:.2f} m/c")
    return line, time_text


movie = FuncAnimation(figure_movie, animate, frames=len(field_frames),
                      interval=30, blit=True)

output_movie = PROJECT_ROOT / "doc" / "fig" / "yuan2010_dgtd_10cm.gif"
movie.save(output_movie, writer=PillowWriter(fps=30))
print(f"saved {output_movie.relative_to(PROJECT_ROOT)} "
      f"({len(field_frames)} frames)")

# Optional: also write an mp4 when ffmpeg is available.
if FFMpegWriter.isAvailable():
    movie.save(output_movie.with_suffix(".mp4"), writer=FFMpegWriter(fps=30))
    print(f"saved {output_movie.with_suffix('.mp4').relative_to(PROJECT_ROOT)}")

plt.close(figure_movie)

# %% [markdown]
# In a notebook the GIF is displayed inline; as a plain script it is written
# to `doc/fig/`.

# %%
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        from IPython.display import Image, display
        display(Image(filename=str(output_movie)))
except ImportError:
    pass

# %% [markdown]
# The DGTD simulation reproduces the analytical reflected power within a few
# tenths of a dB over the whole 2-4 GHz band, including the Fabry-Perot dip
# discussed by Yuan2010.
