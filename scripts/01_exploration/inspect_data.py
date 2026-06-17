# =============================================================================
# inspect_data.py — First look at the PEERS dataset structure
# Explores 1 subject / 1 session to understand file contents and layout.
# =============================================================================

import json
from pathlib import Path

import mne
import pandas as pd

from config import (
    EEG_PATH, EVENTS_PATH, ELEC_PATH, CHAN_PATH,
    BEH_PATH, JSON_PATH, SESSIONS, MONTAGE_NAME
)

# =============================================================================
# Settings — change these to inspect a different subject/session
# =============================================================================
SUB = 'sub-LTP063'
SES = 'ses-0'

# =============================================================================
# 1. Available subjects & sessions on disk
# =============================================================================
print("[ 1 ] AVAILABLE DATA FOLDERS")
print("-" * 40)
folder = Path("./data")
subfolders = sorted([f.name for f in folder.iterdir() if f.is_dir()])
print(", ".join(subfolders))

# =============================================================================
# 2. Load raw EEG and print recording info
# =============================================================================
print("\n[ 2 ] RAW EEG INFO")
print("-" * 40)

raw = mne.io.read_raw_edf(
    EEG_PATH.format(sub=SUB, ses=SES),
    preload=False,
    verbose=False
)

# E129 is the reference electrode, named 'Cz' in electrode files
if 'E129' in raw.ch_names:
    raw.rename_channels({'E129': 'Cz'})

print(raw.info)
print(f"\nDuration:      {raw.times[-1] / 60:.1f} min")
print(f"Channels:      {len(raw.ch_names)}")
print(f"Sampling rate: {raw.info['sfreq']:.0f} Hz")

# =============================================================================
# 3. Channels file
# =============================================================================
print("\n[ 3 ] CHANNELS")
print("-" * 40)

channels = pd.read_csv(CHAN_PATH.format(sub=SUB, ses=SES), sep='\t')
print("Channel types:\n", channels['type'].value_counts())
print("\nChannel status:\n", channels['status'].value_counts())
print("\nEOG channels:\n", channels[channels['type'] == 'EOG']['name'].tolist())
print("\nLast rows:\n", channels.tail())

# =============================================================================
# 4. Events file
# =============================================================================
print("\n[ 4 ] EVENTS")
print("-" * 40)

events = pd.read_csv(EVENTS_PATH.format(sub=SUB, ses=SES), sep='\t')
print("Columns:", events.columns.tolist())
print("Trial types:", events['trial_type'].unique())
print("\nLast 10 rows:\n", events.tail(10).to_string())

# =============================================================================
# 5. Electrodes file
# =============================================================================
print("\n[ 5 ] ELECTRODES")
print("-" * 40)

electrodes = pd.read_csv(ELEC_PATH.format(sub=SUB, ses=SES), sep='\t')
print(electrodes.head())
print(f"\nTotal electrodes: {len(electrodes)}")

# =============================================================================
# 6. EEG JSON sidecar — check montage name across sessions
# =============================================================================
print("\n[ 6 ] MONTAGE NAME PER SESSION")
print("-" * 40)

for ses in SESSIONS:
    path = JSON_PATH.format(sub=SUB, ses=ses)
    try:
        with open(path) as f:
            data = json.load(f)
        montage = data.get('CapManufacturersModelName', 'NOT FOUND')
        print(f"  {ses}: {montage}")
    except FileNotFoundError:
        print(f"  {ses}: file not found")

# =============================================================================
# 7. Behavioral JSON sidecar
# =============================================================================
print("\n[ 7 ] BEHAVIORAL JSON")
print("-" * 40)

beh_path = BEH_PATH.format(sub=SUB, ses=SES)
try:
    with open(beh_path) as f:
        beh = json.load(f)
    print(json.dumps(beh, indent=4))
except FileNotFoundError:
    print("Behavioral JSON not found for this session.")