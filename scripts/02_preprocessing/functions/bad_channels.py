# =============================================================================
# functions/bad_channels.py — Bad channel detection and interpolation
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
    Detects bad channels based on continuous data via three criteria:

      1. Drift     — channel mean > BAD_CH_SD_FACTOR × SD of all means
      2. Amplitude — more than BAD_CH_AMP_FRAC of time points > BAD_CH_AMP_THRESH µV
      3. Variance — channel-SD > BAD_CH_VAR_SD_FACTOR × SD of all channel-SDs

    Protected channels (face, mastoids, Cz) are never returned.

    Returns list of bad channel names.
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
    print(f"    Variance:    {bad_var or '—'}")
    print(f"    Combined:    {combined or '—'}")

    return combined


def interpolate_bad_channels(raw: mne.io.Raw, bad_channels: list[str]) -> None:
    """
    Marks and interpolates bad channels in-place.
    """
    if not bad_channels:
        return

    raw.info['bads'] = bad_channels
    raw.interpolate_bads(reset_bads=True)
    print(f"  Interpolated: {bad_channels}")


def detect_bad_channels_from_epochs(
    epochs: mne.Epochs,
    n_total_trials: int
) -> list[str]:
    """
    Detects bad channels based on epochs that are rejected by
    the amplitude threshold (REJECT_FIRST). A channel is 'bad' if it is the cause of rejection in more
    than BAD_CH_EPOCH_THRESHOLD of the epochs.

    Returns list of channel names that need to be interpolated.
    """
    epochs_tmp = epochs.copy()
    epochs_tmp.drop_bad(reject=REJECT_FIRST)

    all_bad_chs = []
    for entry in epochs_tmp.drop_log:
        all_bad_chs.extend(entry)

    ch_counts = Counter(all_bad_chs)

    print(f"  Most rejected channels (500 µV):")
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