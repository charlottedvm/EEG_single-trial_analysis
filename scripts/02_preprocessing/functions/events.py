# =============================================================================
# functions/events.py -> Load and preprocess events en preprocessen
# =============================================================================

import numpy as np
import pandas as pd

from config import RT_MIN, RT_MAX, MIN_PERFORMANCE, MIN_TRIALS, EVENT_DICT


def preprocess_events(events_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Adds RT correctly to the raw events dataframe.

    Steps:
    - Split stimuli (RECOG_TARGET / RECOG_LURE) and responses (RECOG_RESP)
    - Link each stimulus to the first response within 5 seconds
    - Calculates RT = onset_resp - onset_stimulus
    - Determines correct:
        * TARGET + old_new == 'OLD' → correct
        * LURE   + old_new == 'NEW' → correct

    Returns a copy of the stimulus rows with additional columns: onset_resp, RT, correct.
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
        print("  ⚠ 'old_new' column not found, 'correct' is set to NaN.")

    return merged


def filter_events(events: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """
    Applies performance and RT filters.

    Returns (filtered DataFrame, None) on success,
    or (None, reason) if the session should be skipped.
    """
    performance = events['correct'].mean()
    print(f"  Performance: {performance:.2%}")

    if performance < MIN_PERFORMANCE:
        return None, f"Low performance ({performance:.2%} < {MIN_PERFORMANCE:.0%})"

    total = len(events)
    events_rt = events[(events['RT'] > RT_MIN) & (events['RT'] < RT_MAX)].copy()
    n_removed = total - len(events_rt)
    print(f"  Removed by RT-filter: {n_removed} ({n_removed}/{total})")

    events_filtered = events_rt[events_rt['trial_type'].isin(EVENT_DICT)].copy()
    print(f"  Trials na filtering: {len(events_filtered)}")

    if len(events_filtered) < MIN_TRIALS:
        return None, f"Low number of trials ({len(events_filtered)} < {MIN_TRIALS})"

    return events_filtered, None


def make_mne_events(events: pd.DataFrame, sfreq: float) -> np.ndarray:
    """
    Converts an events DataFrame to an MNE events array (n × 3).
    Column order: [sample, 0, event_id]
    """
    event_samples = (events['onset'] * sfreq).astype(int).values
    event_ids     = events['trial_type'].map(EVENT_DICT).values

    return np.column_stack([
        event_samples,
        np.zeros(len(event_samples), dtype=int),
        event_ids
    ])