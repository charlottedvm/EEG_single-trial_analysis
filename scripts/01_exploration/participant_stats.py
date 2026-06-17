# =============================================================================
# participant_stats.py — Behavioral statistics per participant
# Computes accuracy, reaction times, and trial counts from events files.
# Saves enriched events (with RT + correct columns) to data/behav_data_merged.csv
# =============================================================================

import numpy as np
import pandas as pd

from config import EVENTS_PATH, RT_MIN, RT_MAX

# =============================================================================
# Settings
# =============================================================================
SUB = 'sub-LTP063'
SES = 'ses-0'

# =============================================================================
# 1. Load & preprocess events
# =============================================================================

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


# =============================================================================
# 2. Run for one subject/session
# =============================================================================

events_path = EVENTS_PATH.format(sub=SUB, ses=SES)
events_raw  = pd.read_csv(events_path, sep='\t')
events      = preprocess_events(events_raw)

# Save enriched events
out_path = 'data/behav_data_merged.csv'
events.to_csv(out_path, index=False)
print(f"Saved enriched events to {out_path}\n")

# =============================================================================
# 3. Trial counts
# =============================================================================
print("[ 1 ] TRIAL COUNTS")
print("-" * 40)

total     = len(events)
n_target  = (events['trial_type'] == 'RECOG_TARGET').sum()
n_lure    = (events['trial_type'] == 'RECOG_LURE').sum()
correct   = (events['correct'] == 1).sum()
incorrect = (events['correct'] == 0).sum()

print(f"  Total trials:   {total}")
print(f"  Targets:        {n_target}")
print(f"  Lures:          {n_lure}")
print(f"  Correct:        {correct}")
print(f"  Incorrect:      {incorrect}")

# =============================================================================
# 4. Accuracy
# =============================================================================
print("\n[ 2 ] ACCURACY")
print("-" * 40)

performance = events['correct'].mean()
print(f"  Overall accuracy: {performance:.2%}  {'⚠ LOW (<55%)' if performance < 0.55 else '✓'}")

# Per trial type
for ttype in ['RECOG_TARGET', 'RECOG_LURE']:
    subset = events[events['trial_type'] == ttype]
    acc    = subset['correct'].mean()
    print(f"  {ttype}: {acc:.2%}")

# =============================================================================
# 5. Reaction times
# =============================================================================
print("\n[ 3 ] REACTION TIMES")
print("-" * 40)

rt_vals = events['RT'].dropna()
rt_ok   = events[(events['RT'] > RT_MIN) & (events['RT'] < RT_MAX)]

print(f"  Mean RT:         {rt_vals.mean():.3f} s  (SD = {rt_vals.std():.3f})")
print(f"  Median RT:       {rt_vals.median():.3f} s")
print(f"\n  RT distribution:")
for pct in [5, 25, 50, 75, 95]:
    print(f"    P{pct:2d}: {np.percentile(rt_vals, pct):.3f} s")

n_removed = total - len(rt_ok)
print(f"\n  Removed (RT < {RT_MIN}s or > {RT_MAX}s): {n_removed} trials ({n_removed / total:.1%})")
print(f"  Trials remaining after RT filter:        {len(rt_ok)}")