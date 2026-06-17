import pandas as pd

def preprocess_events(events_raw):
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
        ['trial_type','onset', 'onset_resp', 'RT', 'correct']
    ]

    return events_cleaned