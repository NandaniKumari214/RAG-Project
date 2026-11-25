"""
Laplace Transform and LTI Stability Analysis

Generates symbolic Laplace-transform verifications, a four-regime
pole-zero/ROC dashboard, and a 3D Laplace-magnitude surface.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import scipy.signal as signal
import sympy as sp


OUTPUT_DIR = "figures"

SYSTEMS = [
    {
        "name": "Case 1: Underdamped Stable",
        "transfer": r"$H(s)=\frac{10}{s^2+2s+10}$",
        "num": [10],
        "den": [1, 2, 10],
        "status": "BIBO Stable",
        "color": "green",
    },
    {
        "name": "Case 2: Overdamped Stable",
        "transfer": r"$H(s)=\frac{4}{s^2+5s+4}$",
        "num": [4],
        "den": [1, 5, 4],
        "status": "BIBO Stable",
        "color": "green",
    },
    {
        "name": "Case 3: Marginally Stable",
        "transfer": r"$H(s)=\frac{9}{s^2+9}$",
        "num": [9],
        "den": [1, 0, 9],
        "status": "Marginally Stable",
        "color": "orange",
    },
    {
        "name": "Case 4: Unstable System",
        "transfer": r"$H(s)=\frac{5}{s^2-1.2s+9.36}$",
        "num": [5],
        "den": [1, -1.2, 9.36],
        "status": "Unstable",
        "color": "red",
    },
]


def run_symbolic_verification():
    """Verify five canonical unilateral Laplace-transform pairs with SymPy."""
    t = sp.symbols("t", real=True)
    s = sp.symbols("s", complex=True)
    signals = {
        "Unit Step u(t)": sp.Heaviside(t),
        "Exponential Decay e^(-2t)u(t)": sp.exp(-2 * t) * sp.Heaviside(t),
        "Damped Sine e^(-t)sin(3t)u(t)": (
            sp.exp(-t) * sp.sin(3 * t) * sp.Heaviside(t)
        ),
        "Damped Cosine e^(-2t)cos(4t)u(t)": (
            sp.exp(-2 * t) * sp.cos(4 * t) * sp.Heaviside(t)
        ),
        "Ramp-Modulated Decay t e^(-3t)u(t)": (
            t * sp.exp(-3 * t) * sp.Heaviside(t)
        ),
    }

    print("\nSYMBOLIC UNILATERAL LAPLACE VERIFICATION")
    print("-" * 70)
    for name, expression in signals.items():
        transform, plane, _ = sp.laplace_transform(expression, t, s)
        numerator, denominator = sp.cancel(sp.simplify(transform)).as_numer_denom()
        transform_text = str(
            sp.Mul(
                sp.expand(numerator),
                sp.Pow(sp.expand(denominator), -1, evaluate=False),
                evaluate=False,
            )
        )
        plane_text = str(plane)
        print(
            f"{name:38s} -> X(s) = {transform_text:18s}"
            f" | ROC: Re(s) > {plane_text}"
        )


def plot_stability_regimes():
    """Plot s-plane ROC, impulse response, and step response for four systems."""
    time = np.linspace(0, 8, 1000)
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), constrained_layout=True)
    fig.suptitle(
        "Continuous-Time LTI Stability Regimes",
        fontsize=16,
        fontweight="bold",
    )

    for row, config in enumerate(SYSTEMS):
        system = signal.TransferFunction(config["num"], config["den"])
        zeros, poles, _ = signal.tf2zpk(config["num"], config["den"])
        rightmost_pole = float(np.max(np.real(poles)))

        plane_ax = axes[row, 0]
        plane_ax.set_title(
            f"{config['name']}\n{config['transfer']}  |  {config['status']}",
            fontsize=9,
        )
        plane_ax.axvline(0, color="black", linewidth=1.2)
        plane_ax.axhline(0, color="black", linewidth=0.8)
        plane_ax.add_patch(
            Rectangle(
                (rightmost_pole, -7),
                10,
                14,
                facecolor=config["color"],
                alpha=0.18,
            )
        )
        if len(zeros):
            plane_ax.scatter(
                np.real(zeros),
                np.imag(zeros),
                marker="o",
                s=95,
                facecolors="none",
                edgecolors="blue",
                linewidths=1.5,
                label="Zeros",
            )
        plane_ax.scatter(
            np.real(poles),
            np.imag(poles),
            marker="x",
            s=95,
            color="red",
            linewidths=2,
            label="Poles",
        )
        plane_ax.set_xlim(-5.5, 2.5)
        plane_ax.set_ylim(-6.5, 6.5)
        plane_ax.set_xlabel(r"$\sigma$ (Real axis)")
        plane_ax.set_ylabel(r"$j\omega$ (Imaginary axis)")
        plane_ax.grid(True, alpha=0.25)
        plane_ax.legend(loc="upper right", fontsize=8)

        impulse_time, impulse = signal.impulse(system, T=time)
        axes[row, 1].plot(impulse_time, impulse, color="navy", linewidth=1.5)
        axes[row, 1].set_title("Impulse Response h(t)", fontsize=9)
        axes[row, 1].set_xlabel("Time (s)")
        axes[row, 1].set_ylabel("Amplitude")
        axes[row, 1].grid(True, alpha=0.25)

        step_time, step = signal.step(system, T=time)
        axes[row, 2].plot(step_time, step, color="darkorange", linewidth=1.5)
        axes[row, 2].set_title("Unit Step Response y(t)", fontsize=9)
        axes[row, 2].set_xlabel("Time (s)")
        axes[row, 2].set_ylabel("Amplitude")
        axes[row, 2].grid(True, alpha=0.25)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "laplace_stability_dashboard.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"\nStability dashboard saved to: {output_path}")


def plot_laplace_magnitude_surface():
    """Plot |H(sigma+j omega)| and its Fourier-slice cross-section."""
    num = np.array([10.0])
    den = np.array([1.0, 2.0, 10.0])

    sigma = np.linspace(-3.5, 1.5, 500)
    omega = np.linspace(-6.0, 6.0, 500)
    sigma_grid, omega_grid = np.meshgrid(sigma, omega)
    s = sigma_grid + 1j * omega_grid

    magnitude = np.abs(np.polyval(num, s) / np.polyval(den, s))
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-6))
    magnitude_db = np.clip(magnitude_db, -40, 40)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        sigma_grid,
        omega_grid,
        magnitude_db,
        cmap="viridis",
        edgecolor="none",
        alpha=0.9,
        antialiased=True,
    )
    fig.colorbar(surface, ax=ax, shrink=0.65, label=r"$|H(s)|$ (dB)")

    omega_slice = np.linspace(-6.0, 6.0, 500)
    fourier_magnitude = np.abs(
        np.polyval(num, 1j * omega_slice) / np.polyval(den, 1j * omega_slice)
    )
    fourier_db = np.clip(20 * np.log10(np.maximum(fourier_magnitude, 1e-6)), -40, 40)
    ax.plot(
        np.zeros_like(omega_slice),
        omega_slice,
        fourier_db,
        color="red",
        linewidth=3,
        label=r"$|H(j\omega)|$",
    )

    ax.scatter([-1, -1], [3, -3], [40, 40], color="red", s=60, marker="x")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\omega$")
    ax.set_zlabel(r"$|H(\sigma+j\omega)|$ (dB)")
    ax.set_title("Laplace Magnitude Surface and Fourier Transform Slice")
    ax.set_zlim(-40, 40)
    ax.legend(loc="upper left")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "laplace_magnitude_surface.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"3D Laplace magnitude surface saved to: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_symbolic_verification()
    plot_stability_regimes()
    plot_laplace_magnitude_surface()
    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
