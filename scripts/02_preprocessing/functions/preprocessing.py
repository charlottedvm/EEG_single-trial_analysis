# =============================================================================
# functions/preprocessing.py — Laden, filteren, resamplen, referentie
# =============================================================================

import json

import mne

from config import (
    FACE_CHANNELS, MASTOIDS, MONTAGE_NAME,
    HIGH_PASS, LOW_PASS, NOTCH_FREQ, RESAMPLE
)


def load_and_prepare_raw(raw_path: str, json_path: str) -> mne.io.Raw | None:
    """
    Load an EDF file and perform the basic preprocessing:
      1. Channel E129 rename to Cz
      2. Mount setup (only HydroCel-129 supported)
      3. Face channels and Cz drop
      4. Load into memory
      5. Notch-filter (50 Hz)
      6. Bandwidth-filter (0.1 - 30 Hz)
      7. Resampling to 256 Hz
      8. Re-reference linked mastoids (E57 and E100)

    Returns the preprocessed Raw object, or None if the montage is not recognized.
    """
    raw = mne.io.read_raw_edf(raw_path, preload=False, verbose=False)

    # E129 is the reference electrode, is named 'Cz' in the electrode files
    if 'E129' in raw.ch_names:
        raw.rename_channels({'E129': 'Cz'})

    # Montage
    with open(json_path) as f:
        eeg_json = json.load(f)

    montage_name = eeg_json.get('CapManufacturersModelName', '')
    if 'HydroCel' not in montage_name:
        print(f"  ⚠ unknown montage: '{montage_name}', skip session.")
        return None

    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    raw.set_montage(montage, match_case=False, on_missing='ignore')
    print(f"  Montage: {MONTAGE_NAME}")

    # Drop face channels (inclusive Cz — SD = 0 after re-referencing)
    to_drop = [ch for ch in FACE_CHANNELS + ['Cz'] if ch in raw.ch_names]
    raw.drop_channels(to_drop)
    print(f"  Dropped: {len(to_drop)} channels (face + Cz)")

    # Filter, resampling en re-referencing
    raw.load_data()
    raw.notch_filter(freqs=NOTCH_FREQ, picks='eeg', verbose=False)
    raw.filter(HIGH_PASS, LOW_PASS, verbose=False)
    raw.resample(RESAMPLE, npad='auto')
    raw.set_eeg_reference(ref_channels=MASTOIDS, verbose=False)
    print(f"  Filtered ({HIGH_PASS}–{LOW_PASS} Hz), resampled to {RESAMPLE} Hz, ref → mastoids")

    return raw