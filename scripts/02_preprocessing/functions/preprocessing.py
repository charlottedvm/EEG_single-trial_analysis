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
    Laadt een EDF-bestand en voert de basispreprocessing uit:
      1. Kanaal E129 hernoemen naar Cz
      2. Montage instellen (alleen HydroCel-129 ondersteund)
      3. Gezichtskanalen en Cz droppen
      4. Laden in geheugen
      5. Notch-filter (50 Hz)
      6. Bandbreedte-filter (HIGH_PASS – LOW_PASS Hz)
      7. Resamplen naar RESAMPLE Hz
      8. Re-referentie naar mastoïden

    Returns het preprocessed Raw object, of None als de montage niet herkend wordt.
    """
    raw = mne.io.read_raw_edf(raw_path, preload=False, verbose=False)

    # E129 is de referentie-elektrode, heet 'Cz' in de elektroden-bestanden
    if 'E129' in raw.ch_names:
        raw.rename_channels({'E129': 'Cz'})

    # Montage
    with open(json_path) as f:
        eeg_json = json.load(f)

    montage_name = eeg_json.get('CapManufacturersModelName', '')
    if 'HydroCel' not in montage_name:
        print(f"  ⚠ Onbekende montage: '{montage_name}', sessie overgeslagen.")
        return None

    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    raw.set_montage(montage, match_case=False, on_missing='ignore')
    print(f"  Montage: {MONTAGE_NAME}")

    # Gezichtskanalen droppen (inclusief Cz — SD = 0 na re-referencing)
    to_drop = [ch for ch in FACE_CHANNELS + ['Cz'] if ch in raw.ch_names]
    raw.drop_channels(to_drop)
    print(f"  Gedropt: {len(to_drop)} kanalen (gezicht + Cz)")

    # Filteren, resamplen en re-referentie
    raw.load_data()
    raw.notch_filter(freqs=NOTCH_FREQ, picks='eeg', verbose=False)
    raw.filter(HIGH_PASS, LOW_PASS, verbose=False)
    raw.resample(RESAMPLE, npad='auto')
    raw.set_eeg_reference(ref_channels=MASTOIDS, verbose=False)
    print(f"  Gefilterd ({HIGH_PASS}–{LOW_PASS} Hz), geresamplet naar {RESAMPLE} Hz, ref → mastoïden")

    return raw