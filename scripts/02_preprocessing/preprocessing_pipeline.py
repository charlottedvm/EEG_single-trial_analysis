# =============================================================================
# pipeline.py — Hoofd-preprocessingscript voor de PEERS EEG dataset
#
# Steps per session:
#   1. Load data + montage + filter + resample + re-reference
#   2. Load events + RT-filter + performance check
#   3. Detect bad channels on continuous data + interpolation
#   4. Fit ICA + ICLabel
#   5. Create epochs
#   6. Detect bad channels on epochs + re-interpolation (if needed)
#   7. Epochs cleaning: 500 µV → apply ICA → baseline correction → 200 µV
#   8. ERP sanity check (FN400 / LPC)
#   9. Save as .fif
# =============================================================================

import gc
from pathlib import Path

import config as cfg
from functions import (
    preprocess_events,
    load_and_prepare_raw,
    detect_bad_channels,
    interpolate_bad_channels,
    detect_bad_channels_from_epochs,
    run_ica,
    make_epochs,
    run_sanity_checks,
)
from functions.events import filter_events, make_mne_events
from functions.epochs import make_epochs


# =============================================================================
# Main loop
# =============================================================================
for sub in cfg.get_subjects():
    for ses in cfg.get_sessions(sub):

        print(f"\n{'='*60}")
        print(f"  Processing: {sub} | {ses}")
        print(f"{'='*60}")

        # Skip if already processed
        out = cfg.out_file(sub, ses)
        if out.exists() and not cfg.FORCE_REPROCESS:
            print(f"  Already processed, skipped: {out}")
            continue

        raw_p    = cfg.raw_path(sub, ses)
        events_p = cfg.events_path(sub, ses)
        json_p   = cfg.json_path(sub, ses)

        if not Path(raw_p).exists():
            print(f"  ⚠ File not found, skipped: {raw_p}")
            continue

        # ------------------------------------------------------------------
        # Stap 1 — Load data, montage, filter, resample, re-reference
        # ------------------------------------------------------------------
        raw = load_and_prepare_raw(raw_p, json_p)
        if raw is None:
            continue

        # ------------------------------------------------------------------
        # Stap 2 — Events
        # ------------------------------------------------------------------
        import pandas as pd
        events_raw = pd.read_csv(events_p, sep='\t')
        events     = preprocess_events(events_raw)

        events, skip_reason = filter_events(events)
        if skip_reason:
            print(f"  ⚠ Session skipped: {skip_reason}")
            del raw
            gc.collect()
            continue

        events_mne = make_mne_events(events, raw.info['sfreq'])

        # ------------------------------------------------------------------
        # Stap 3 — Bad channels on continuous data
        # ------------------------------------------------------------------
        initial_bad = detect_bad_channels(raw)
        interpolate_bad_channels(raw, initial_bad)

        # ------------------------------------------------------------------
        # Stap 4 — ICA
        # ------------------------------------------------------------------
        ica = run_ica(raw)

        # ------------------------------------------------------------------
        # Stap 5 — Create epochs for epoch-based bad channel check
        # ------------------------------------------------------------------
        import mne, numpy as np
        from config import EPOCH_TMIN, EPOCH_TMAX, REJECT_FIRST

        epochs_tmp = mne.Epochs(
            raw,
            events_mne,
            event_id={'target': 1, 'lure': 2},
            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
            baseline=None,
            preload=True,
            metadata=events.reset_index(drop=True),
            verbose=False
        )
        n_total = len(epochs_tmp)

        # ------------------------------------------------------------------
        # Stap 6 — Bad channels on epochs + re-interpolation
        # ------------------------------------------------------------------
        epoch_bad = detect_bad_channels_from_epochs(epochs_tmp, n_total)
        del epochs_tmp
        gc.collect()

        if epoch_bad:
            print(f"  Extra slechte kanalen via epochs: {epoch_bad}")
            interpolate_bad_channels(raw, epoch_bad)

        # ------------------------------------------------------------------
        # Stap 7 — Definitive epochs + cleaning (500 µV → ICA → 200 µV)
        # ------------------------------------------------------------------
        epochs = make_epochs(raw, events_mne, events, ica)

        if epochs is None:
            print(f"  ⚠ No epochs left, session skipped.")
            del raw, ica
            gc.collect()
            continue

        n_final = len(epochs)
        if n_final < 0.5 * n_total:
            print(f"  ⚠ Less than 50% of trials left ({n_final}/{n_total})")

        print(f"\n  Final epochs: {n_final}")

        # ------------------------------------------------------------------
        # Stap 8 — ERP sanity check
        # ------------------------------------------------------------------
        # sanity = run_sanity_checks(epochs, sub, ses)
        # if not sanity.get('pass', False):
        #     print(f"  ⚠ ERP sanity check failed -> check data.")

        # ------------------------------------------------------------------
        # Stap 9 — Opslaan
        # ------------------------------------------------------------------
        epochs.save(out, overwrite=True)
        print(f"\n  Saved: {out}")

        del raw, epochs, ica
        gc.collect()

print("\n\nDone.")