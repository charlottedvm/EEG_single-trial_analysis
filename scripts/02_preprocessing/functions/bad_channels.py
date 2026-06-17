# =============================================================================
# functions/bad_channels.py — Bad channel detectie en interpolatie
# =============================================================================

from collections import Counter

import mne
import numpy as np

from config import (
    PROTECTED_CHANNELS,
    BAD_CH_SD_FACTOR, BAD_CH_AMP_THRESH, BAD_CH_AMP_FRAC,
    BAD_CH_VAR_SD_FACTOR, BAD_CH_EPOCH_THRESHOLD,
    REJECT_FIRST
)


def detect_bad_channels(raw: mne.io.Raw) -> list[str]:
    """
    Detecteert slechte kanalen op basis van continue data via drie criteria:

      1. Drift     — kanaalgemiddelde > BAD_CH_SD_FACTOR × SD van alle gemiddelden
      2. Amplitude — meer dan BAD_CH_AMP_FRAC van de tijdstappen > BAD_CH_AMP_THRESH µV
      3. Variantie — kanaal-SD > BAD_CH_VAR_SD_FACTOR × SD van alle kanaal-SDs

    Beschermde kanalen (gezicht, mastoïden, Cz) worden nooit teruggegeven.

    Returns lijst van slechte kanaalnamen.
    """
    data_arr    = raw.get_data(picks='eeg')
    eeg_picks   = mne.pick_types(raw.info, eeg=True)
    eeg_names   = [raw.ch_names[i] for i in eeg_picks]

    # Criterium 1: drift
    ch_means = np.mean(data_arr, axis=1)
    bad_sd   = [
        eeg_names[i] for i, m in enumerate(ch_means)
        if np.abs(m - np.mean(ch_means)) > BAD_CH_SD_FACTOR * np.std(ch_means)
    ]

    # Criterium 2: amplitude
    bad_amp = [
        eeg_names[i] for i, ch in enumerate(data_arr)
        if np.mean(np.abs(ch) > BAD_CH_AMP_THRESH) > BAD_CH_AMP_FRAC
    ]

    # Criterium 3: variantie
    ch_std  = np.std(data_arr, axis=1)
    bad_var = [
        eeg_names[i] for i, s in enumerate(ch_std)
        if s > np.mean(ch_std) + BAD_CH_VAR_SD_FACTOR * np.std(ch_std)
    ]

    combined = list(set(bad_sd + bad_amp + bad_var))
    combined = [ch for ch in combined if ch not in PROTECTED_CHANNELS]

    print(f"  Bad channels (continue data):")
    print(f"    Drift:       {bad_sd or '—'}")
    print(f"    Amplitude:   {bad_amp or '—'}")
    print(f"    Variantie:   {bad_var or '—'}")
    print(f"    Gecombineerd: {combined or '—'}")

    return combined


def interpolate_bad_channels(raw: mne.io.Raw, bad_channels: list[str]) -> None:
    """
    Markeert en interpoleert slechte kanalen in-place.
    """
    if not bad_channels:
        return

    raw.info['bads'] = bad_channels
    raw.interpolate_bads(reset_bads=True)
    print(f"  Geïnterpoleerd: {bad_channels}")


def detect_bad_channels_from_epochs(
    epochs: mne.Epochs,
    n_total_trials: int
) -> list[str]:
    """
    Detecteert slechte kanalen op basis van epochs die afgekeurd worden door
    de amplitude-drempel (REJECT_FIRST). Een kanaal is 'bad' als het in meer
    dan BAD_CH_EPOCH_THRESHOLD van de epochs de oorzaak van afkeuring is.

    Returns lijst van kanaalnamen die geïnterpoleerd moeten worden.
    """
    epochs_tmp = epochs.copy()
    epochs_tmp.drop_bad(reject=REJECT_FIRST)

    all_bad_chs = []
    for entry in epochs_tmp.drop_log:
        all_bad_chs.extend(entry)

    ch_counts = Counter(all_bad_chs)

    print(f"  Meest afgewezen kanalen (500 µV):")
    for ch, count in ch_counts.most_common(10):
        pct = count / n_total_trials * 100
        print(f"    {ch}: {count} epochs ({pct:.1f}%)")

    epoch_bad = [
        ch for ch, count in ch_counts.items()
        if count > n_total_trials * BAD_CH_EPOCH_THRESHOLD
        and ch not in PROTECTED_CHANNELS
    ]

    del epochs_tmp
    return epoch_bad