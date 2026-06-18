# =============================================================================
# functions/ica_cleaning.py — ICA fitting en ICLabel-gebaseerde rejectie
# =============================================================================

import warnings
from collections import Counter

import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

from config import (
    ICA_N_COMPONENTS, ICA_METHOD, ICA_RANDOM_STATE, ICA_MAX_ITER,
    ICA_EYE_PROB, ICA_HIGH_PASS, ICA_LOW_PASS
)


def run_ica(raw: mne.io.Raw) -> ICA:
    """
    Fits ICA on a broad band-pass filtered copy of the data (1-100 Hz for ICLabel),
    labels components with ICLabel, and marks eye-components to remove.

    The ICA is then ready to be applied to epochs via ica.apply(epochs).
    The raw data is not modified.

    Returns the fitted and configured ICA object.
    """
    # Aparte kopie voor ICA-fitting (ICLabel vereist 1–100 Hz)
    raw_ica = raw.copy().filter(ICA_HIGH_PASS, ICA_LOW_PASS, verbose=False)

    # Slechte segmenten annoteren (vermijdt dat grote artefacten de ICA domineren)
    mne.preprocessing.annotate_amplitude(
        raw_ica,
        peak=500e-6,
        flat=1e-6,
        picks='eeg',
        verbose=False
    )

    n_bad_seg = sum(1 for ann in raw_ica.annotations if ann['description'].startswith('BAD'))
    print(f"  Bad segments for ICA: {n_bad_seg}")

    # ICA fitten
    ica = ICA(
        n_components=ICA_N_COMPONENTS,
        method=ICA_METHOD,
        fit_params=dict(extended=True),
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ica.fit(raw_ica, verbose=False)

    print(f"  ICA: {ica.n_components_} components found")

    # Labelen met ICLabel
    ic_labels = label_components(raw_ica, ica, method='iclabel')

    _print_ica_summary(ic_labels)

    # Oogcomponenten met hoge betrouwbaarheid markeren
    eye_idx = [
        i for i, (lab, prob) in enumerate(
            zip(ic_labels['labels'], ic_labels['y_pred_proba'])
        )
        if 'eye' in lab.lower() and prob > ICA_EYE_PROB
    ]

    print(f"\n  Components removed: {eye_idx}")
    ica.exclude = eye_idx

    del raw_ica
    return ica


def _print_ica_summary(ic_labels: dict) -> None:
    """Print a summary of ICA labels and sanity checks."""
    print("\n" + "=" * 40)
    print("ICA component labels:")
    print("=" * 40)

    for i, (lab, prob) in enumerate(
        zip(ic_labels['labels'], ic_labels['y_pred_proba'])
    ):
        print(f"  IC {i:02d}: {lab:10s} | probability = {prob:.2f}")

    label_counts = Counter(ic_labels['labels'])
    print("\n  Summary:")
    for k, v in label_counts.items():
        print(f"    {k}: {v}")

    # Sanity checks
    print("\n  Sanity checks:")
    n_eye = sum(1 for lab in ic_labels['labels'] if 'eye' in lab.lower())
    if n_eye == 0:
        print("    ⚠ No eye components detected")

    if len(label_counts) == 1:
        print("    ⚠ All components have the same label")

    low_conf = [p for p in ic_labels['y_pred_proba'] if p < 0.6]
    if len(low_conf) > len(ic_labels['y_pred_proba']) * 0.5:
        print("    ⚠ More than half of the classifications have low reliability")

    print("=" * 40 + "\n")