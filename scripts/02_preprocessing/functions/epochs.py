# =============================================================================
# functions/epochs.py — Epochs aanmaken en opschonen
# =============================================================================

import gc

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA

from config import (
    EPOCH_TMIN, EPOCH_TMAX, BASELINE,
    REJECT_FIRST, REJECT_FINAL
)


def make_epochs(
    raw: mne.io.Raw,
    events_mne: np.ndarray,
    events_df: pd.DataFrame,
    ica: ICA,
) -> mne.Epochs | None:
    """
    Maakt epochs aan en voert een drietraps opschoonprocedure uit:

      Stap 1 — Eerste amplitude-rejectie (500 µV) om erge artefacten te verwijderen
               vóór ICA wordt toegepast. Dit wordt ook gebruikt om slechte kanalen
               per epoch te detecteren (zie bad_channels.detect_bad_channels_from_epochs).

      Stap 2 — ICA toepassen (oog-componenten verwijderen)

      Stap 3 — Baseline correctie + finale amplitude-rejectie (200 µV)

    Returns het finale Epochs object, of None als er geen epochs overblijven.

    Note: stap 2 (bad channel detectie op epochs + herinterpolatie) zit in
    pipeline.py zodat raw.interpolate_bads() toegepast kan worden vóór het
    opnieuw aanmaken van epochs.
    """
    event_id = {'target': 1, 'lure': 2}

    def _create(raw_obj: mne.io.Raw) -> mne.Epochs:
        return mne.Epochs(
            raw_obj,
            events_mne,
            event_id=event_id,
            tmin=EPOCH_TMIN,
            tmax=EPOCH_TMAX,
            baseline=None,          # baseline later toepassen, na ICA
            preload=True,
            metadata=events_df.reset_index(drop=True),
            verbose=False
        )

    epochs = _create(raw)
    print(f"  Epochs voor opschonen: {len(epochs)}")

    # Stap 1 — Eerste rejectie (500 µV)
    epochs.drop_bad(reject=REJECT_FIRST)
    print(f"  Epochs na 500 µV rejectie: {len(epochs)}")

    if len(epochs) == 0:
        print("  ⚠ Geen epochs over na eerste rejectie.")
        return None

    # Stap 2 — ICA toepassen
    ica.apply(epochs)
    print(f"  ICA toegepast ({len(ica.exclude)} componenten verwijderd)")

    # Stap 3 — Baseline + finale rejectie (200 µV)
    epochs.apply_baseline(BASELINE)
    epochs.drop_bad(reject=REJECT_FINAL)
    print(f"  Epochs na 200 µV rejectie (finaal): {len(epochs)}")

    if len(epochs) == 0:
        print("  ⚠ Geen epochs over na finale rejectie.")
        return None

    return epochs