# =============================================================================
# quality_checks.py — EEG data quality check
# Run BEFORE the preprocessing pipeline to flag problematic sessions.
# =============================================================================

import json

import mne
import numpy as np
import pandas as pd

from config import (
    EEG_PATH, EVENTS_PATH, JSON_PATH, ELEC_PATH,
    FACE_CHANNELS, MONTAGE_NAME,
    RT_MIN, RT_MAX,
    BAD_CH_MEAN_SD_FACTOR, BAD_CH_AMP_THRESH_UV, BAD_CH_VAR_SD_FACTOR
)
from participant_stats import preprocess_events

# =============================================================================
# Settings — change to check a different subject/session
# =============================================================================
SUB = 'sub-LTP063'
SES = 'ses-2'

# =============================================================================
# Paths
# =============================================================================
raw_path    = EEG_PATH.format(sub=SUB, ses=SES)
events_path = EVENTS_PATH.format(sub=SUB, ses=SES)
json_path   = JSON_PATH.format(sub=SUB, ses=SES)

print(f"\n{'='*60}")
print(f"  EEG QUALITY CHECK — {SUB} | {SES}")
print(f"{'='*60}\n")

# =============================================================================
# 1. Recording info
# =============================================================================
print("[ 1 ] RECORDING INFO")
print("-" * 40)

raw = mne.io.read_raw_edf(raw_path, preload=False, verbose=False)

if 'E129' in raw.ch_names:
    raw.rename_channels({'E129': 'Cz'})

duration_min = raw.times[-1] / 60
n_channels   = len(raw.ch_names)
sfreq        = raw.info['sfreq']

print(f"  Recording duration: {duration_min:.1f} min ({raw.times[-1]:.0f} sec)")
print(f"  Channels (total):   {n_channels}")
print(f"  Sampling rate:      {sfreq:.0f} Hz")
print(f"  E129/Cz present:    {'Yes' if 'Cz' in raw.ch_names else 'No'}")

with open(json_path) as f:
    eeg_json = json.load(f)
montage_name = eeg_json.get('CapManufacturersModelName', 'UNKNOWN')
supported    = 'HydroCel' in montage_name or 'Geodesic' in montage_name
print(f"  Montage name:       {montage_name}")
print(f"  Montage supported:  {'Yes (HydroCel-129)' if supported else 'NO — CHECK!'}")

# =============================================================================
# 2. Events & behavior
# =============================================================================
print("\n[ 2 ] EVENTS & BEHAVIOR")
print("-" * 40)

events_raw  = pd.read_csv(events_path, sep='\t')
events      = preprocess_events(events_raw)

total        = len(events)
n_target     = (events['trial_type'] == 'RECOG_TARGET').sum()
n_lure       = (events['trial_type'] == 'RECOG_LURE').sum()
performance  = events['correct'].mean()
rt_mean      = events['RT'].mean()
rt_std       = events['RT'].std()
rt_ok        = events[(events['RT'] > RT_MIN) & (events['RT'] < RT_MAX)]
n_rt_removed = total - len(rt_ok)

print(f"  Total trials:       {total}")
print(f"  Targets:            {n_target}")
print(f"  Lures:              {n_lure}")
print(f"  Accuracy:           {performance:.1%}  {'⚠ LOW (<55%)' if performance < 0.55 else '✓'}")
print(f"  Mean RT:            {rt_mean:.3f} s (SD = {rt_std:.3f})")
print(f"  Removed (RT):       {n_rt_removed} trials ({n_rt_removed / total:.1%})")
print(f"  Trials after RT:    {len(rt_ok)}")

rt_vals = events['RT'].dropna()
print(f"\n  RT distribution:")
for pct in [5, 25, 50, 75, 95]:
    print(f"    P{pct:2d}: {np.percentile(rt_vals, pct):.3f} s")

# =============================================================================
# 3. Raw amplitude check
# =============================================================================
print("\n[ 3 ] AMPLITUDE CHECK (raw, before filtering)")
print("-" * 40)

montage = mne.channels.make_standard_montage(MONTAGE_NAME)
raw.set_montage(montage, match_case=False, on_missing='ignore')
raw.drop_channels([ch for ch in FACE_CHANNELS if ch in raw.ch_names])
raw.load_data()

data      = raw.get_data(picks='eeg')
eeg_picks = mne.pick_types(raw.info, eeg=True)
eeg_names = [raw.ch_names[i] for i in eeg_picks]

global_max  = np.max(np.abs(data)) * 1e6
global_mean = np.mean(np.abs(data)) * 1e6
global_std  = np.std(data) * 1e6
pct95       = np.percentile(np.abs(data), 95) * 1e6
pct99       = np.percentile(np.abs(data), 99) * 1e6

print(f"  Global max:         {global_max:.1f} µV")
print(f"  Global mean (|x|):  {global_mean:.2f} µV")
print(f"  Global std:         {global_std:.2f} µV")
print(f"  95th percentile:    {pct95:.1f} µV")
print(f"  99th percentile:    {pct99:.1f} µV")

if global_max > 5000:
    print("\n  ⚠ Extremely high amplitude — likely DC drift or detached electrode")
elif global_max > 1000:
    print("\n  ⚠ High amplitudes present — check bad channels")
else:
    print("\n  ✓ Amplitudes within expected range")

ch_maxes  = np.max(np.abs(data), axis=1) * 1e6
top10_idx = np.argsort(ch_maxes)[::-1][:10]
print("\n  Top 10 channels (highest peak):")
for idx in top10_idx:
    print(f"    {eeg_names[idx]:6s}: {ch_maxes[idx]:.1f} µV")

for thresh in [200, 500, 1000]:
    pct_above  = np.mean(np.abs(data) > thresh * 1e-6) * 100
    n_ch_above = np.sum(np.mean(np.abs(data) > thresh * 1e-6, axis=1) > 0.05)
    print(f"\n  > {thresh} µV: {pct_above:.2f}% of samples | {n_ch_above} ch >5% of the time")

# =============================================================================
# 4. Bad channel estimate
# =============================================================================
print("\n[ 4 ] BAD CHANNEL ESTIMATE")
print("-" * 40)

ch_means = np.mean(data, axis=1)
bad_sd   = [eeg_names[i] for i, m in enumerate(ch_means)
            if np.abs(m - np.mean(ch_means)) > BAD_CH_MEAN_SD_FACTOR * np.std(ch_means)]

bad_amp  = [eeg_names[i] for i, ch in enumerate(data)
            if np.mean(np.abs(ch) > BAD_CH_AMP_THRESH_UV * 1e-6) > 0.20]

ch_std   = np.std(data, axis=1)
bad_var  = [eeg_names[i] for i, s in enumerate(ch_std)
            if s > np.mean(ch_std) + BAD_CH_VAR_SD_FACTOR * np.std(ch_std)]

all_bad  = list(set(bad_sd + bad_amp + bad_var))

print(f"  SD-based:           {bad_sd or '—'}")
print(f"  Amplitude (500 µV): {bad_amp or '—'}")
print(f"  Variance (3 SD):    {bad_var or '—'}")
print(f"  Combined:           {all_bad or '—'}  ({len(all_bad)} channels)")

if len(all_bad) > 10:
    print(f"\n  ⚠ Many bad channels ({len(all_bad)}) — consider skipping session")
elif len(all_bad) > 5:
    print(f"\n  ⚠ Several bad channels — check data quality")
else:
    print(f"\n  ✓ Bad channel count acceptable")

# =============================================================================
# 5. Channel variance
# =============================================================================
print("\n[ 5 ] CHANNEL VARIANCE")
print("-" * 40)

ch_std_uv = ch_std * 1e6
print(f"  Mean SD across channels:  {np.mean(ch_std_uv):.2f} µV")
print(f"  Min SD (quietest):        {np.min(ch_std_uv):.2f} µV  ({eeg_names[np.argmin(ch_std)]})")
print(f"  Max SD (noisiest):        {np.max(ch_std_uv):.2f} µV  ({eeg_names[np.argmax(ch_std)]})")

flat_channels = [eeg_names[i] for i, s in enumerate(ch_std)
                 if s < 1e-7 and eeg_names[i] != 'Cz']
if flat_channels:
    print(f"\n  ⚠ Flat channels (possibly detached): {flat_channels}")
else:
    print(f"\n  ✓ No flat channels found")

# =============================================================================
# 6. Summary
# =============================================================================
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")

checks = [
    ("Montage recognized",    supported),
    ("Accuracy >= 55%",       performance >= 0.55),
    ("Amplitudes < 1000 µV",  global_max < 1000),
    ("Bad channels <= 5",     len(all_bad) <= 5),
    ("No flat channels",      len(flat_channels) == 0),
    ("Trials after RT >= 50", len(rt_ok) >= 50),
]

for name, ok in checks:
    print(f"  {'✓' if ok else '✗'}  {name}")

n_pass = sum(ok for _, ok in checks)
print(f"\n  Score: {n_pass}/{len(checks)}")
if n_pass == len(checks):
    print("  → Data looks good, proceed with preprocessing.")
elif n_pass >= 4:
    print("  → Data usable, but check the failed items above.")
else:
    print("  → ⚠ Multiple issues — consider skipping this session.")

print(f"\n{'='*60}\n")