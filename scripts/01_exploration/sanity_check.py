# =============================================================================
# erp_sanity_check.py — Electrode layout sanity check
# Visualizes the GSN-HydroCel-129 montage and identifies face / EOG channels.
# Run this once to verify that coordinate-based channel removal is correct.
# =============================================================================

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import (
    ELEC_PATH, EOG_CHANNELS, FACE_CHANNELS,
    MONTAGE_NAME, Z_THRESH, Y_THRESH
)

# =============================================================================
# Settings
# =============================================================================
SUB = 'sub-LTP063'
SES = 'ses-0'

# =============================================================================
# 1. Load electrode coordinates from file (subject-specific)
# =============================================================================
electrodes = pd.read_csv(ELEC_PATH.format(sub=SUB, ses=SES), sep='\t')

# =============================================================================
# 2. Derive face channels from MNE standard montage (for reference)
# =============================================================================
montage = mne.channels.make_standard_montage(MONTAGE_NAME)
pos     = montage.get_positions()['ch_pos']

face_from_montage = sorted([
    ch for ch, xyz in pos.items()
    if xyz[2] < Z_THRESH and xyz[1] > Y_THRESH
])
face_no_eog = [ch for ch in face_from_montage if ch not in EOG_CHANNELS]

print("[ CHANNEL IDENTIFICATION ]")
print("-" * 40)
print(f"Face channels (coord-based, incl. EOG): {len(face_from_montage)} → {face_from_montage}")
print(f"EOG channels:                            {len(EOG_CHANNELS)} → {EOG_CHANNELS}")
print(f"Bad channels (face excl. EOG):           {len(face_no_eog)} → {face_no_eog}")
print(f"\nExpected from paper (Weidemann & Kahana): 26 face channels")
print(f"Found with z < {Z_THRESH} and y > {Y_THRESH}:          {len(face_from_montage)}")

# =============================================================================
# 3. Helper: color per electrode
# =============================================================================
def get_color(row):
    if row['name'] in FACE_CHANNELS:
        return 'red'
    if row['name'] in EOG_CHANNELS:
        return 'green'
    return 'steelblue'

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='EEG'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',       markersize=8, label='Face (remove)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green',     markersize=8, label='EOG'),
]

# =============================================================================
# 4. Plot: subject-specific electrode coordinates (3 views)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'Electrode layout — {SUB} {SES}', fontsize=13)

ax1, ax2, ax3 = axes

# Top view (x / y)
for _, row in electrodes.iterrows():
    c = get_color(row)
    ax1.scatter(row['x'], row['y'], color=c, s=40)
    ax1.text(row['x'], row['y'] + 0.002, row['name'], fontsize=7, ha='center')
ax1.axhline(y=Y_THRESH, color='orange', linestyle='--', linewidth=1, label=f'y={Y_THRESH}')
ax1.set_title('Top view (x / y)\nnose = top')
ax1.set_xlabel('x (left / right)')
ax1.set_ylabel('y (front / back)')
ax1.set_aspect('equal')
ax1.legend(handles=legend_elements, fontsize=7)

# Side view (y / z)
for _, row in electrodes.iterrows():
    c = get_color(row)
    ax2.scatter(row['y'], row['z'], color=c, s=40)
    ax2.text(row['y'], row['z'] + 0.002, row['name'], fontsize=7, ha='center')
ax2.axhline(y=Z_THRESH, color='orange', linestyle='--', linewidth=1, label=f'z={Z_THRESH}')
ax2.axvline(x=Y_THRESH, color='purple', linestyle='--', linewidth=1, label=f'y={Y_THRESH}')
ax2.set_title('Side view (y / z)\nnose = right')
ax2.set_xlabel('y (front / back)')
ax2.set_ylabel('z (up / down)')
ax2.set_aspect('equal')
ax2.legend(handles=legend_elements, fontsize=7)

# 3D view
ax3.remove()
ax3 = fig.add_subplot(133, projection='3d')
for _, row in electrodes.iterrows():
    c = get_color(row)
    ax3.scatter(row['x'], row['y'], row['z'], color=c, s=30)
ax3.set_title('3D view')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')

plt.tight_layout()
plt.savefig('data/electrode_layout_subject.png', dpi=150)
plt.show()
print("\nSaved: data/electrode_layout_subject.png")

# =============================================================================
# 5. Plot: MNE standard montage (for comparison)
# =============================================================================
montage.plot(show=True)

# =============================================================================
# 6. Verify channel names match between EDF and electrodes file
# =============================================================================
print("\n[ CHANNEL NAME CHECK ]")
print("-" * 40)
print("Note: E129 in EDF = Cz in electrodes file → rename before processing.")

elec_names = set(electrodes['name'].tolist())
eeg_names  = set([f'E{i}' for i in range(1, 130)] + ['Cz'])  # approximate

missing_in_elec = [ch for ch in ['E129'] if ch not in elec_names]
if missing_in_elec:
    print(f"⚠ These EDF channel names are absent from the electrodes file: {missing_in_elec}")
    print("  → Rename E129 → Cz before setting montage.")
else:
    print("✓ Channel names appear consistent.")