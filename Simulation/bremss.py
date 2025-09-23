import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# --- Bethe-Heitler approximations ---
def brems_spectrum(k, Emax, t_over_X0):
    y = k / Emax
    return (t_over_X0 / k) * (4/3 - 4/3 * y + y**2)

def pair_share(x):
    return 1.0 - (4.0/3.0) * x * (1 - x)

# --- Differential distribution for E+ ---
def pdf_Eplus(Ep, Emax, t_over_X0):
    if Ep <= 0 or Ep >= Emax:
        return 0.0

    def integrand(k):
        x = Ep / k
        return brems_spectrum(k, Emax, t_over_X0) * pair_share(x) / k

    val, _ = quad(integrand, Ep, Emax, epsabs=1e-8, epsrel=1e-6)
    return val

# --- Differential distribution for E- (same as E+) ---
def pdf_Eminus(Em, E0, t_over_X0):
    return pdf_Eplus(Em, E0, t_over_X0)

# --- Build normalized PDFs ---
def build_pdfs(Emin,Emax, X0=8.897, t_cm=0.01, npts=400):
    t_over_X0 = t_cm / X0
    E_vals = np.logspace(np.log10(Emin), np.log10(Emax), npts)

    # Photon spectrum (normalized)
    photon_vals = np.array([brems_spectrum(k, Emax, t_over_X0) for k in E_vals])
    photon_vals /= np.trapezoid(photon_vals, E_vals)

    # e+ spectrum
    eplus_vals = np.array([pdf_Eplus(E, Emax, t_over_X0) for E in E_vals])
    eplus_vals /= np.trapezoid(eplus_vals, E_vals)

    return E_vals, photon_vals, eplus_vals

# --- Correct sampling on bins ---
def sample_from_pdf_on_bins(E_vals, pdf_vals, nsamples=10000, rng=None, sample_log_within_bin=False):
    if rng is None:
        rng = np.random.default_rng()

    # construct edges
    edges = np.zeros(len(E_vals) + 1)
    edges[1:-1] = np.sqrt(E_vals[:-1] * E_vals[1:])  # geometric mean
    edges[0] = E_vals[0]**2 / edges[1]
    edges[-1] = E_vals[-1]**2 / edges[-2]

    bin_widths = edges[1:] - edges[:-1]
    p_bin = pdf_vals * bin_widths
    p_bin = np.maximum(p_bin, 0.0)
    p_bin /= p_bin.sum()

    cdf = np.cumsum(p_bin)
    rnd = rng.random(nsamples)
    bin_indices = np.searchsorted(cdf, rnd, side='right')
    bin_indices = np.clip(bin_indices, 0, len(E_vals)-1)

    u = rng.random(nsamples)
    samples = np.empty(nsamples)
    for i, b in enumerate(bin_indices):
        lo, hi = edges[b], edges[b+1]
        if sample_log_within_bin:
            samples[i] = np.exp(np.log(lo) + u[i] * (np.log(hi) - np.log(lo)))
        else:
            samples[i] = lo + u[i] * (hi - lo)
    return samples

# --- Histogram normalized to density ---
def normalized_hist_density(samples, bins):
    counts, edges = np.histogram(samples, bins=bins, density=False)
    bin_widths = np.diff(edges)
    N = samples.size
    density = counts / (N * bin_widths)   # probability per GeV
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers, density, edges


if __name__ == "__main__":
    X0 = 8.897  # cm
    t_cm = 0.01 # cm = 100 µm
    XoverX0 = t_cm/X0
    Emin = 0.01 # GeV
    # Emin = 0.5 # GeV
    Emax = 10 # GeV

    E_vals, photons, eplus = build_pdfs(Emin,Emax, X0, t_cm)

    nsamples = 1000000
    rng = np.random.default_rng(123)
    sampled_photons = sample_from_pdf_on_bins(E_vals, photons, nsamples, rng)
    sampled_eplus   = sample_from_pdf_on_bins(E_vals, eplus, nsamples, rng)

    # bins = np.logspace(np.log10(E_vals[0]), np.log10(E0), 100)
    bins = np.linspace(E_vals[0], Emax, 400) # linear bins

    for samples, label, color in [
        (sampled_photons, "γ samples", "C0"),
        (sampled_eplus,   "e+ samples", "C1"),
    ]:
        centers, density, _ = normalized_hist_density(samples, bins)
        plt.step(centers, density, where="mid", label=label, color=color)

    # Overlay analytic PDFs
    plt.plot(E_vals, photons, "--", lw=1.5, color="C0", label="γ PDF")
    plt.plot(E_vals, eplus,   "--", lw=1.5, color="C1", label="e+ PDF")

    # plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Probability density [1/GeV]")
    plt.title("Sampled spectra with analytic PDFs (10 GeV e⁻ on 100 µm Al)")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.show()