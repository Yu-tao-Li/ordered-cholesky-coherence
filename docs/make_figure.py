"""Draw docs/admissible_disc.png: the disc of admissible c23, the coordinate repair and the three phase classes."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BLUE, RED, GREEN, ORANGE, GREY = '#0072B2', '#D55E00', '#009E73', '#E69F00', '#555555'


def disc(alpha, beta):
    return np.conj(alpha) * beta, np.sqrt((1 - abs(alpha) ** 2) * (1 - abs(beta) ** 2))


def frame(ax, title):
    ax.set_aspect('equal')
    ax.axhline(0, color='#bbbbbb', lw=.6, zorder=0)
    ax.axvline(0, color='#bbbbbb', lw=.6, zorder=0)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls=':', color=GREY, lw=.8))
    ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), xlabel=r'Re $c_{23}$', ylabel=r'Im $c_{23}$')
    ax.set_title(title, fontsize=10, loc='left')


fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), dpi=150)
theta = np.linspace(0, 2 * np.pi, 400)

# (a) for c12 = alpha and c13 = beta, c23 must lie in a disc; the coordinate repair projects onto it radially
ax = axes[0]
frame(ax, '(a) Admissible $c_{23}$ and the coordinate repair')
alpha, beta = 0.8, 0.75 * np.exp(1j * np.pi / 3)
centre, radius = disc(alpha, beta)
ax.add_patch(plt.Circle((centre.real, centre.imag), radius, color=BLUE, alpha=.18, lw=0))
ax.add_patch(plt.Circle((centre.real, centre.imag), radius, fill=False, color=BLUE, lw=1.2))
ax.plot(centre.real, centre.imag, '+', color=BLUE, ms=8)
ax.annotate('centre $\\bar{\\alpha}\\beta$\nradius $\\sqrt{(1-|\\alpha|^2)(1-|\\beta|^2)}$', (centre.real, centre.imag),
            xytext=(-1.08, .82), fontsize=8.5, color=BLUE, arrowprops=dict(arrowstyle='-', color=BLUE, lw=.6))
candidate = -0.45 - 0.35j
repaired = centre + (candidate - centre) * radius / abs(candidate - centre)
ax.plot(candidate.real, candidate.imag, 'x', color=RED, ms=8, mew=2)
ax.annotate('', (repaired.real, repaired.imag), (candidate.real, candidate.imag),
            arrowprops=dict(arrowstyle='->', color=GREY, lw=1.2))
ax.plot(repaired.real, repaired.imag, 'o', color=BLUE, ms=6)
ax.text(candidate.real - .05, candidate.imag - .17, 'pairwise value\n(inadmissible)', fontsize=8.5, color=RED, ha='center', va='top')
ax.text(repaired.real + .06, repaired.imag - .02, 'repaired', fontsize=8.5, color=BLUE, va='top')
ax.text(-1.1, -1.1, r'$|\alpha|=0.8,\ |\beta|=0.75$; $c_{12},c_{13}$ unchanged', fontsize=8, color=GREY)

# (b) a magnitude |c23| = h is admissible for every phase, some phases or no phase
ax = axes[1]
frame(ax, '(b) Phase classes of a magnitude $|c_{23}|$')
alpha, beta = 0.9, 0.1
centre, radius = disc(alpha, beta)
ax.add_patch(plt.Circle((centre.real, centre.imag), radius, color=BLUE, alpha=.18, lw=0))
ax.add_patch(plt.Circle((centre.real, centre.imag), radius, fill=False, color=BLUE, lw=1.2))
ab = abs(centre)
for h, color, label in [(.5 * (radius - ab), GREEN, 'every phase'), (radius, ORANGE, 'some phases'),
                        (.5 * (radius + ab + 1), RED, 'no phase')]:
    ax.plot(h * np.cos(theta), h * np.sin(theta), color=color, lw=1.4, label=f'$|c_{{23}}|={h:.2f}$: {label}')
ax.legend(loc='lower left', fontsize=8, frameon=False)
ax.text(-1.1, 1.0, r'$|\alpha|=0.9,\ |\beta|=0.1$', fontsize=8, color=GREY)

fig.tight_layout()
fig.savefig(HERE / 'admissible_disc.png', facecolor='white')
print('saved', HERE / 'admissible_disc.png')
