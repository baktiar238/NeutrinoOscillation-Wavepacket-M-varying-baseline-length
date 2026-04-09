import numpy as np
import matplotlib.pyplot as plt

# ── Mixing angles ─────────────────────────────────────────────────────────────
t12 = np.deg2rad(33.41)
t13 = np.deg2rad(8.54)
t23 = np.deg2rad(49.1)

# ── Wave-packet parameters ────────────────────────────────────────────────────
# sigma : production coherence length in km
# eps   : fractional energy spread (dimensionless), set 0 for monochromatic beam
sigma = 1e-4   # km  (~0.1 m, typical for accelerator neutrinos)
eps   = 0.0    # dimensionless

# ── Mass-squared differences (eV²) ───────────────────────────────────────────
delta12 = 7.42e-5   # Δm²₁₂  solar
delta13 = 2.51e-3   # Δm²₁₃  atmospheric, normal ordering
delta23 = - delta13 + delta12

# ── Baseline and energy ───────────────────────────────────────────────────────
L    = np.linspace(0, 13000, 50000)   # km  (start at 0.1 to avoid log(0))
E    = 5.0                              # GeV
E_eV = E * 1e9                         # eV

# ── Unit conversion ───────────────────────────────────────────────────────────
# 1 km = 5.06773e9 eV^{-1}   (hbar*c = 197.3 MeV·fm, 1 fm = 1e-18 km)
km_to_iEV = 5.06773e9          # multiply km → eV^{-1}

# L and sigma both converted to eV^{-1} once, used everywhere
L_iEV    = L     * km_to_iEV   # shape (50000,)  eV^{-1}
sigma_iEV = sigma * km_to_iEV  # scalar          eV^{-1}

# ── Matter potential ──────────────────────────────────────────────────────────
Ye = 0.5

def fn_rho(l):
    """Step-model Earth density along baseline (g/cm³)."""
    if   l < 260:          return 2.70
    elif l < 520:          return 2.75
    elif l < 780:          return 2.85
    elif l < 1040:         return 2.90
    else:                  return 2.70

# ── Vacuum PMNS matrix ────────────────────────────────────────────────────────
def U_matrix(CP, t12, t13, t23):
    s12, s23, s13 = np.sin(t12), np.sin(t23), np.sin(t13)
    c12, c23, c13 = np.cos(t12), np.cos(t23), np.cos(t13)
    eid = np.exp(1j * CP)
    return np.array([
        [ c12*c13,                          s12*c13,                          s13/eid          ],
        [-s12*c23 - c12*s23*s13*eid,         c12*c23 - s12*s23*s13*eid,         s23*c13          ],
        [ s12*s23 - c12*c23*s13*eid,        -c12*s23 - s12*c23*s13*eid,         c23*c13          ]
    ], dtype=complex)

# ── Matter-diagonalisation ────────────────────────────────────────────────────
def get_W_and_lambdas(CP, t12, t13, t23, E_eV, rho, Ye):
    """
    Returns W  (3×3 matter mixing matrix, W[flavour, mass_index])
    and     lambdas  (effective Hamiltonian eigenvalues in eV, sorted ascending).

    H is built in eV (divided by 2E already so eigenvalues ~ delta_m²/2E scale).
    """
    U = U_matrix(CP, t12, t13, t23)

    # Vacuum Hamiltonian in flavour basis  [eV]
    # We set the lightest eigenvalue to 0 (overall phase irrelevant)
    H_vac = U @ np.diag([0.0,
                          delta12 / (2.0 * E_eV),
                          delta13 / (2.0 * E_eV)]) @ U.conj().T

    # Matter potential  V = sqrt(2) G_F N_e  [eV]
    V = 7.56e-14 * rho * Ye   # eV

    H_matter = H_vac + np.diag([V, 0.0, 0.0])

    # eigh returns SORTED (ascending) eigenvalues and orthonormal eigenvectors
    lambdas, W = np.linalg.eigh(H_matter)
    # W[:, i] = i-th eigenvector  →  W[flavour, mass_index]
    return W, lambdas   # lambdas in eV

# ── Average density (constant-density approximation) ─────────────────────────
rho_avg = 2.848   # g/cm³  DUNE-like average over 1300 km

# ── CP phases to plot ─────────────────────────────────────────────────────────
CP_degrees = [180]
CP_phases  = np.deg2rad(CP_degrees)

# ── Main loop ─────────────────────────────────────────────────────────────────
for CP in CP_phases:

    W, lambdas = get_W_and_lambdas(CP, t12, t13, t23, E_eV, rho_avg, Ye)

    # Effective eigenvalue differences  [eV]
    dl_01 = lambdas[1] - lambdas[0]
    dl_12 = lambdas[2] - lambdas[1]
    dl_02 = lambdas[2] - lambdas[0]

    # ── Oscillation phases  [dimensionless] ──────────────────────────────────
    # phase = (lambda_j - lambda_i) [eV]  *  L [eV^{-1}]
    phase_01 = dl_01 * L_iEV
    phase_12 = dl_12 * L_iEV
    phase_02 = dl_02 * L_iEV

    # ── Oscillation lengths  [eV^{-1}] ───────────────────────────────────────
    # L_osc defined so that phase = 2π  →  L_osc = 2π / dl
    L_osc_01 = 2.0 * np.pi / dl_01
    L_osc_12 = 2.0 * np.pi / dl_12
    L_osc_02 = 2.0 * np.pi / dl_02

    # ── Coherence lengths  [eV^{-1}] ─────────────────────────────────────────
    # L_coh = sqrt(2) * E² * sigma / dl²
    # all quantities in eV / eV^{-1}
    L_coh_01 = 4 * np.sqrt(2.0) * E_eV**2 * sigma_iEV / dl_01**2
    L_coh_12 = 4 * np.sqrt(2.0) * E_eV**2 * sigma_iEV / dl_12**2
    L_coh_02 = 4 * np.sqrt(2.0) * E_eV**2 * sigma_iEV / dl_02**2

    # ── Damping exponentials (vectorised over L_iEV) ──────────────────────────
    # Both terms dimensionless: (L/L_coh)² and (sigma/L_osc)²
    def damp(L_coh, L_osc):
        return np.exp(-(L_iEV / L_coh)**2
                      - 2.0 * np.pi**2 * eps**2 * (sigma_iEV / L_osc)**2)

    exp_01 = damp(L_coh_01, L_osc_01)
    exp_12 = damp(L_coh_12, L_osc_12)
    exp_02 = damp(L_coh_02, L_osc_02)

    # ── W-matrix combinations  (scalars) ─────────────────────────────────────
    # For  P(νe → νμ)  [alpha=e=0, beta=μ=1]
    W1_mu = W[1,1].conj() * W[0,1] * W[1,0] * W[0,0].conj()   # (i,j)=(0,1)
    W2_mu = W[1,2].conj() * W[0,2] * W[1,1] * W[0,1].conj()   # (i,j)=(1,2)
    W3_mu = W[1,2].conj() * W[0,2] * W[1,0] * W[0,0].conj()   # (i,j)=(0,2)

    # For  P(νe → ντ)  [alpha=e=0, beta=τ=2]
    W1_ta = W[1,1].conj() * W[2,1] * W[1,0] * W[2,0].conj()
    W2_ta = W[1,2].conj() * W[2,2] * W[1,1] * W[2,1].conj()
    W3_ta = W[1,2].conj() * W[2,2] * W[1,0] * W[2,0].conj()

    # ── Constant (averaged) terms ─────────────────────────────────────────────
    const_mu = (  (W[1,0]*W[1,0].conj() * W[0,0]*W[0,0].conj())
                + (W[1,1]*W[1,1].conj() * W[0,1]*W[0,1].conj())
                + (W[1,2]*W[1,2].conj() * W[0,2]*W[0,2].conj()) ).real

    const_ta = (  (W[1,0]*W[1,0].conj() * W[2,0]*W[2,0].conj())
                + (W[1,1]*W[1,1].conj() * W[2,1]*W[2,1].conj())
                + (W[1,2]*W[1,2].conj() * W[2,2]*W[2,2].conj()) ).real

    # ── P(νe → νμ) ────────────────────────────────────────────────────────────
    P_emu = (
        const_mu
        + 2*(  W1_mu.real * np.cos(phase_01) * exp_01
             + W2_mu.real * np.cos(phase_12) * exp_12
             + W3_mu.real * np.cos(phase_02) * exp_02)
        + 2*(  W1_mu.imag * np.sin(phase_01) * exp_01
             + W2_mu.imag * np.sin(phase_12) * exp_12
             + W3_mu.imag * np.sin(phase_02) * exp_02)
    )

    # ── P(νe → ντ) ────────────────────────────────────────────────────────────
    P_eta = (
        const_ta
        + 2*(  W1_ta.real * np.cos(phase_01) * exp_01
             + W2_ta.real * np.cos(phase_12) * exp_12
             + W3_ta.real * np.cos(phase_02) * exp_02)
        + 2*(  W1_ta.imag * np.sin(phase_01) * exp_01
             + W2_ta.imag * np.sin(phase_12) * exp_12
             + W3_ta.imag * np.sin(phase_02) * exp_02)
    )

    # ── P(νe → νe) survival ───────────────────────────────────────────────────
    P_ee = 1.0 - P_emu - P_eta

    # ── Sanity-check: clip tiny numerical noise ───────────────────────────────
    P_emu = np.clip(P_emu.real, 0, 1)
    P_eta = np.clip(P_eta.real, 0, 1)
    P_ee  = np.clip(P_ee.real,  0, 1)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.ylim(-0.05, 1.05)
    plt.xscale('log')
    plt.xlabel(r"$L$ (km)", fontsize=13)
    plt.ylabel("Probability", fontsize=13)
    plt.plot(L, P_emu, label=r"$\nu_{\mu} \rightarrow \nu_{e}$",  lw=1.5)
    plt.plot(L, P_ee,  label=r"$\nu_{\mu} \rightarrow \nu_{\mu}$",    lw=1.5)
    plt.plot(L, P_eta, label=r"$\nu_{\mu} \rightarrow \nu_{\tau}$", lw=1.5)
    plt.title(rf"CP = {np.rad2deg(CP):.0f}° | Matter (Wave-Packet) | $E$ = {E} GeV",
              fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

print("Done. All plots saved.")
