# =============================================================================
# functions/events.py -> Load and preprocess events en preprocessen
# =============================================================================

import numpy as np
import pandas as pd

from config import RT_MIN, RT_MAX, MIN_PERFORMANCE, MIN_TRIALS, EVENT_DICT


def preprocess_events(events_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Merge stimulus and response rows and calculate reaction times (RT).

    Parameters
    ----------
    events_raw : pd.DataFrame
        Raw events TSV loaded from BIDS dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned events with:
        - onset
        - RT
        - correct (1 = correct, 0 = incorrect)
    """

    # Select recognition stimuli
    stimuli = events_raw[
        events_raw['trial_type'].isin(['RECOG_TARGET', 'RECOG_LURE'])
    ].copy()

    # Select only real recognition responses
    responses = events_raw[
        events_raw['trial_type'] == 'RECOG_RESP'
    ][['onset']].copy()

    # Sort by onset
    stimuli = stimuli.sort_values('onset').reset_index(drop=True)
    responses = responses.sort_values('onset').reset_index(drop=True)

    # Rename response onset column
    responses = responses.rename(columns={'onset': 'onset_resp'})

    # Match each stimulus to the next response within 5 seconds
    events_cleaned = pd.merge_asof(
        stimuli,
        responses,
        left_on='onset',
        right_on='onset_resp',
        direction='forward',
        tolerance=5.0
    )

    # Calculate reaction time
    events_cleaned['RT'] = (
        events_cleaned['onset_resp'] - events_cleaned['onset']
    )

    # Remove trials without responses
    n_before = len(events_cleaned)

    events_cleaned = events_cleaned.dropna(
        subset=['onset_resp']
    ).reset_index(drop=True)

    n_dropped = n_before - len(events_cleaned)

    if n_dropped > 0:
        print(f"Trials zonder response geskipt: {n_dropped}")

    # Determine correctness
    events_cleaned['correct'] = (
        (
            (events_cleaned['trial_type'] == 'RECOG_TARGET') &
            (events_cleaned['recog_resp'] == 1)
        ) |
        (
            (events_cleaned['trial_type'] == 'RECOG_LURE') &
            (events_cleaned['recog_resp'] == 0)
        )
    ).astype(int)

    # Keep only relevant columns
    events_cleaned = events_cleaned[
        ['trial_type', 'onset', 'onset_resp', 'RT', 'correct']
    ]

    return events_cleaned


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