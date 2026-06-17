# =============================================================================
# functions/sanity_check.py — ERP sanity checks: FN400 en LPC
#
# Verwachte effecten op basis van Weidemann & Kahana (herkenningsgeheugen):
#   FN400 (300–500 ms, frontaal) — oud > nieuw (positiever voor targets)
#   LPC   (500–800 ms, pariëtaal) — oud > nieuw (positiever voor targets)
#
# Relevante kanalen (GSN-HydroCel-129):
#   Frontaal  → E5, E6, E7, E12, E13, E106, E112
#   Pariëtaal → E52, E53, E54, E60, E61, E62, E67, E72, E77, E79
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import mne
from torch import sub

from config import DERIVATIVES_DIR

# Tijdvensters en kanalen
FN400_WINDOW   = (0.300, 0.500)    # sec
LPC_WINDOW     = (0.500, 0.800)    # sec
FRONTAL_CHS    = ['E5', 'E6', 'E7', 'E12', 'E13', 'E106', 'E112']
PARIETAL_CHS   = ['E52', 'E53', 'E54', 'E60', 'E61', 'E62', 'E67', 'E72', 'E77', 'E79']
MIN_TRIALS_PER_COND = 10


def run_sanity_checks(epochs: mne.Epochs, sub: str, ses: str) -> dict:
    """
    Berekent FN400 en LPC amplitudes voor targets en lures en controleert
    of de verwachte richtingen aanwezig zijn.

    Verwacht dat epochs de condities 'target' en 'lure' bevatten en
    dat er al een baseline-correctie is toegepast.

    Returns een dict met resultaten en flags:
      {
        'fn400_target': float,   # µV
        'fn400_lure':   float,
        'fn400_ok':     bool,    # target > lure?
        'lpc_target':   float,
        'lpc_lure':     float,
        'lpc_ok':       bool,
        'n_target':     int,
        'n_lure':       int,
        'pass':         bool,    # beide effecten in verwachte richting?
      }
    """
    results = {}

    # Condities ophalen
    targets = epochs['target'] if 'target' in epochs.event_id else None
    lures   = epochs['lure']   if 'lure'   in epochs.event_id else None

    n_target = len(targets) if targets else 0
    n_lure   = len(lures)   if lures   else 0
    results['n_target'] = n_target
    results['n_lure']   = n_lure

    print(f"\n[ SANITY CHECK — {sub} | {ses} ]")
    print(f"  Trials: {n_target} targets | {n_lure} lures")

    if n_target < MIN_TRIALS_PER_COND or n_lure < MIN_TRIALS_PER_COND:
        print(f"  ⚠ Te weinig trials voor betrouwbare ERP-check (min {MIN_TRIALS_PER_COND})")
        results['pass'] = False
        return results

    # Gemiddelde ERPs
    evoked_target = targets.average()
    evoked_lure   = lures.average()

    # Helperfunctie: gemiddeld amplitude in tijdvenster over kanalenset
    def _mean_amp(evoked: mne.Evoked, channels: list[str], tmin: float, tmax: float) -> float:
        picks = [evoked.ch_names.index(ch) for ch in channels if ch in evoked.ch_names]
        if not picks:
            return np.nan
        times = evoked.times
        mask  = (times >= tmin) & (times <= tmax)
        return float(evoked.data[picks][:, mask].mean() * 1e6)   # → µV

    # FN400
    fn400_target = _mean_amp(evoked_target, FRONTAL_CHS,  *FN400_WINDOW)
    fn400_lure   = _mean_amp(evoked_lure,   FRONTAL_CHS,  *FN400_WINDOW)
    fn400_diff   = fn400_target - fn400_lure
    fn400_ok     = fn400_diff > 0

    # LPC
    lpc_target = _mean_amp(evoked_target, PARIETAL_CHS, *LPC_WINDOW)
    lpc_lure   = _mean_amp(evoked_lure,   PARIETAL_CHS, *LPC_WINDOW)
    lpc_diff   = lpc_target - lpc_lure
    lpc_ok     = lpc_diff > 0

    results.update({
        'fn400_target': fn400_target,
        'fn400_lure':   fn400_lure,
        'fn400_ok':     fn400_ok,
        'lpc_target':   lpc_target,
        'lpc_lure':     lpc_lure,
        'lpc_ok':       lpc_ok,
        'pass':         fn400_ok and lpc_ok,
    })

    # Afdrukken
    fn400_sym = '✓' if fn400_ok else '⚠'
    lpc_sym   = '✓' if lpc_ok   else '⚠'

    print(f"\n  FN400 ({FN400_WINDOW[0]*1000:.0f}–{FN400_WINDOW[1]*1000:.0f} ms, frontaal):")
    print(f"    Target: {fn400_target:+.2f} µV | Lure: {fn400_lure:+.2f} µV | Δ = {fn400_diff:+.2f} µV  {fn400_sym}")

    print(f"\n  LPC ({LPC_WINDOW[0]*1000:.0f}–{LPC_WINDOW[1]*1000:.0f} ms, pariëtaal):")
    print(f"    Target: {lpc_target:+.2f} µV | Lure: {lpc_lure:+.2f} µV | Δ = {lpc_diff:+.2f} µV  {lpc_sym}")

    if results['pass']:
        print("\n  ✓ Beide ERP-effecten in verwachte richting.")
    else:
        missed = []
        if not fn400_ok:
            missed.append('FN400')
        if not lpc_ok:
            missed.append('LPC')
        print(f"\n  ⚠ Ontbrekende effecten: {', '.join(missed)} — controleer data.")

    # Plot
    _plot_erps(evoked_target, evoked_lure, sub, ses)

    return results


def _plot_erps(
    evoked_target: mne.Evoked,
    evoked_lure:   mne.Evoked,
    sub: str,
    ses: str,
) -> None:
    """
    Maakt een overzichtsplot met frontale en pariëtale ERP-golven
    voor targets en lures.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle(f'ERP sanity check — {sub} | {ses}', fontsize=12)

    for ax, channels, label, window in zip(
        axes,
        [FRONTAL_CHS, PARIETAL_CHS],
        ['Frontaal (FN400)', 'Pariëtaal (LPC)'],
        [FN400_WINDOW, LPC_WINDOW],
    ):
        for evoked, color, name in [
            (evoked_target, 'steelblue', 'Target'),
            (evoked_lure,   'tomato',    'Lure'),
        ]:
            picks = [evoked.ch_names.index(ch) for ch in channels if ch in evoked.ch_names]
            if not picks:
                continue
            mean_wave = evoked.data[picks].mean(axis=0) * 1e6
            ax.plot(evoked.times * 1000, mean_wave, label=name, color=color, linewidth=1.5)

        # Venster markeren
        ax.axvspan(window[0] * 1000, window[1] * 1000, alpha=0.12, color='gray')
        ax.axhline(0, color='k', linewidth=0.6)
        ax.axvline(0, color='k', linewidth=0.6, linestyle='--')
        ax.set_xlabel('Tijd (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.invert_yaxis()   # EEG-conventie: negatief naar boven

    plt.tight_layout()
    save_path = DERIVATIVES_DIR / f"{sub}_{ses}_erp_sanity.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Plot opgeslagen: {save_path}")