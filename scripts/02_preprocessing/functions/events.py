# =============================================================================
# functions/events.py — Events laden en preprocessen
# =============================================================================

import numpy as np
import pandas as pd

from config import RT_MIN, RT_MAX, MIN_PERFORMANCE, MIN_TRIALS, EVENT_DICT


def preprocess_events(events_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Voegt RT en correct toe aan een ruwe events DataFrame.

    Stappen:
    - Splits stimuli (RECOG_TARGET / RECOG_LURE) en responses (RECOG_RESP)
    - Koppelt elke stimulus aan de eerstvolgende respons binnen 5 seconden
    - Berekent RT = onset_resp - onset_stimulus
    - Bepaalt correct:
        * TARGET + old_new == 'OLD' → correct
        * LURE   + old_new == 'NEW' → correct

    Returns een kopie van de stimulusrijen met extra kolommen: onset_resp, RT, correct.
    """
    stimuli   = events_raw[events_raw['trial_type'].isin(['RECOG_TARGET', 'RECOG_LURE'])].copy()
    responses = events_raw[events_raw['trial_type'] == 'RECOG_RESP'][['onset']].copy()

    stimuli   = stimuli.sort_values('onset').reset_index(drop=True)
    responses = responses.sort_values('onset').reset_index(drop=True)
    responses = responses.rename(columns={'onset': 'onset_resp'})

    merged = pd.merge_asof(
        stimuli,
        responses,
        left_on='onset',
        right_on='onset_resp',
        direction='forward',
        tolerance=5.0
    )

    merged['RT'] = merged['onset_resp'] - merged['onset']

    if 'old_new' in merged.columns:
        merged['correct'] = (
            ((merged['trial_type'] == 'RECOG_TARGET') & (merged['old_new'] == 'OLD')) |
            ((merged['trial_type'] == 'RECOG_LURE')   & (merged['old_new'] == 'NEW'))
        ).astype(int)
    else:
        merged['correct'] = np.nan
        print("  ⚠ 'old_new' kolom niet gevonden — 'correct' staat op NaN.")

    return merged


def filter_events(events: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """
    Past performance- en RT-filters toe.

    Returns (gefilterde DataFrame, None) bij succes,
    of (None, reden) als de sessie overgeslagen moet worden.
    """
    performance = events['correct'].mean()
    print(f"  Performance: {performance:.2%}")

    if performance < MIN_PERFORMANCE:
        return None, f"Lage performance ({performance:.2%} < {MIN_PERFORMANCE:.0%})"

    total = len(events)
    events_rt = events[(events['RT'] > RT_MIN) & (events['RT'] < RT_MAX)].copy()
    n_removed = total - len(events_rt)
    print(f"  Verwijderd door RT-filter: {n_removed} ({n_removed}/{total})")

    events_filtered = events_rt[events_rt['trial_type'].isin(EVENT_DICT)].copy()
    print(f"  Trials na filtering: {len(events_filtered)}")

    if len(events_filtered) < MIN_TRIALS:
        return None, f"Te weinig trials ({len(events_filtered)} < {MIN_TRIALS})"

    return events_filtered, None


def make_mne_events(events: pd.DataFrame, sfreq: float) -> np.ndarray:
    """
    Zet een events DataFrame om naar een MNE events array (n × 3).
    Kolomvolgorde: [sample, 0, event_id]
    """
    event_samples = (events['onset'] * sfreq).astype(int).values
    event_ids     = events['trial_type'].map(EVENT_DICT).values

    return np.column_stack([
        event_samples,
        np.zeros(len(event_samples), dtype=int),
        event_ids
    ])