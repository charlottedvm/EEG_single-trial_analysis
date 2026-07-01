import mne
from pathlib import Path

epo_file = Path("data/preprocessed/sub-LTP063_ses-0_epo.fif")  # pad naar je _epo.fif bestand

epochs = mne.read_epochs(epo_file, preload=True, verbose=False)

# 1. Algemene info (kanalen, sfreq, etc.)
print(epochs.info)

# 2. De metadata-tabel (per epoch/trial, bv. condition, RT, etc.)
print(epochs.metadata)

# 3. Kolomnamen van de metadata
if epochs.metadata is not None:
    print(epochs.metadata.columns.tolist())
else:
    print("Geen metadata aanwezig in dit bestand.")

# 4. Eerste paar rijen bekijken
print(epochs.metadata.head())

# 5. Aantal trials per conditie/event
print(epochs.metadata["event_name"].value_counts() if "event_name" in epochs.metadata.columns else epochs.event_id)