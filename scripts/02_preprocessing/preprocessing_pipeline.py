# =============================================================================
# pipeline.py — Hoofd-preprocessingscript voor de PEERS EEG dataset
#
# Volgorde per sessie:
#   1. Laden + montage + filteren + resamplen + re-referentie
#   2. Events laden + RT-filter + performance check
#   3. Bad channel detectie op continue data + interpolatie
#   4. ICA fitten + ICLabel
#   5. Epochs aanmaken
#   6. Bad channel detectie op epochs + herinterpolatie (indien nodig)
#   7. Epochs opschonen: 500 µV → ICA toepassen → baseline → 200 µV
#   8. ERP sanity check (FN400 / LPC)
#   9. Opslaan als .fif
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
# Hoofd-loop
# =============================================================================
for sub in cfg.get_subjects():
    for ses in cfg.get_sessions(sub):

        print(f"\n{'='*60}")
        print(f"  Verwerken: {sub} | {ses}")
        print(f"{'='*60}")

        # Sla over als al verwerkt
        out = cfg.out_file(sub, ses)
        if out.exists() and not cfg.FORCE_REPROCESS:
            print(f"  Al verwerkt, overgeslagen: {out}")
            continue

        raw_p    = cfg.raw_path(sub, ses)
        events_p = cfg.events_path(sub, ses)
        json_p   = cfg.json_path(sub, ses)

        if not Path(raw_p).exists():
            print(f"  ⚠ Bestand niet gevonden, overgeslagen: {raw_p}")
            continue

        # ------------------------------------------------------------------
        # Stap 1 — Laden, montage, filteren, resamplen, re-referentie
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
            print(f"  ⚠ Sessie overgeslagen: {skip_reason}")
            del raw
            gc.collect()
            continue

        events_mne = make_mne_events(events, raw.info['sfreq'])

        # ------------------------------------------------------------------
        # Stap 3 — Bad channels op continue data
        # ------------------------------------------------------------------
        initial_bad = detect_bad_channels(raw)
        interpolate_bad_channels(raw, initial_bad)

        # ------------------------------------------------------------------
        # Stap 4 — ICA
        # ------------------------------------------------------------------
        ica = run_ica(raw)

        # ------------------------------------------------------------------
        # Stap 5 — Eerste epochs aanmaken voor epoch-gebaseerde bad channel check
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
        # Stap 6 — Bad channels op epochs + herinterpolatie
        # ------------------------------------------------------------------
        epoch_bad = detect_bad_channels_from_epochs(epochs_tmp, n_total)
        del epochs_tmp
        gc.collect()

        if epoch_bad:
            print(f"  Extra slechte kanalen via epochs: {epoch_bad}")
            interpolate_bad_channels(raw, epoch_bad)

        # ------------------------------------------------------------------
        # Stap 7 — Definitieve epochs + opschonen (500 µV → ICA → 200 µV)
        # ------------------------------------------------------------------
        epochs = make_epochs(raw, events_mne, events, ica)

        if epochs is None:
            print(f"  ⚠ Geen epochs over, sessie overgeslagen.")
            del raw, ica
            gc.collect()
            continue

        n_final = len(epochs)
        if n_final < 0.5 * n_total:
            print(f"  ⚠ Minder dan 50% van trials over ({n_final}/{n_total})")

        print(f"\n  Final epochs: {n_final}")

        # ------------------------------------------------------------------
        # Stap 8 — ERP sanity check
        # ------------------------------------------------------------------
        sanity = run_sanity_checks(epochs, sub, ses)
        if not sanity.get('pass', False):
            print(f"  ⚠ ERP sanity check niet geslaagd — data controleren.")

        # ------------------------------------------------------------------
        # Stap 9 — Opslaan
        # ------------------------------------------------------------------
        epochs.save(out, overwrite=True)
        print(f"\n  Saved: {out}")

        del raw, epochs, ica
        gc.collect()

print("\n\nKlaar.")