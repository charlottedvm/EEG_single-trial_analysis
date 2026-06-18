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
    Creates epochs and performs a three-step cleaning procedure:

      Step 1 — First amplitude rejection (500 µV) to remove severe artifacts
               before ICA is applied. This is also used to detect bad channels
               per epoch (see bad_channels.detect_bad_channels_from_epochs).

      Step 2 — Apply ICA (remove eye components)

      Step 3 — Baseline correction + final amplitude rejection (200 µV)

    Returns the final Epochs object, or None if no epochs remain.

    Note: step 2 (bad channel detection on epochs + reinterpolation) is in
    pipeline.py so that raw.interpolate_bads() can be applied before
    recreating epochs.
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

    # Step 1 — First rejection (500 µV)
    epochs.drop_bad(reject=REJECT_FIRST)
    print(f"  Epochs na 500 µV rejectie: {len(epochs)}")

    if len(epochs) == 0:
        print("  ⚠ Geen epochs over na eerste rejectie.")
        return None

    # Step 2 — Apply ICA
    ica.apply(epochs)
    print(f"  ICA applied ({len(ica.exclude)} components removed)")

    # Step 3 — Baseline + final rejection (200 µV)
    epochs.apply_baseline(BASELINE)
    epochs.drop_bad(reject=REJECT_FINAL)
    print(f"  Epochs  after 200 µV rejection (finaal): {len(epochs)}")

    if len(epochs) == 0:
        print("  ⚠ No epochs left after finale rejoction.")
        return None

    return epochs