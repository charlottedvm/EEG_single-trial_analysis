# =====================================================================
# THESIS ANALYSE-UITBREIDING
#
# Vereist dat de volgende variabelen al bestaan in je sessie:
#   - epoch_data      (hmp xarray dataset, uit hmp.io.read_mne_data)
#   - estimates       (hmp xarray output, uit model.fit_transform)
#   - hmp_features    (pandas DataFrame met single-trial HMP latencies)
#   - times           (pandas DataFrame met event durations + metadata)
#   - group_erps      (dict: conditie -> grand-average mne.Evoked)
#   - subject_erps    (dict: conditie -> lijst van mne.Evoked per subject)
#   - difference      (mne.Evoked, Target - Lure)
#   - info            (mne.Info, kanaalinformatie)
#
# Structuur (volgt je Resultaten-hoofdstuk):
#   1. Reliability            -> split-half, Spearman-Brown, ICC
#   2. Neurophysiological     -> statistische toetsen op ERP/HMP plausibiliteit
#   3. Functional utility     -> SVM, within-subject CV, ROC, AUC
#   4. Pseudo-trial analyse   -> AUC als functie van SNR (2,4,8,16 trials)
# =====================================================================

import numpy as np
import pandas as pd
import xarray as xr
import mne
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import mne
import hmp
import os
import xarray as xr

from pathlib import Path
from mne.io import read_info

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print(
        "pingouin niet gevonden -> ICC wordt overgeslagen.\n"
        "Installeer met: pip install pingouin"
    )
# =====================================================================
# VARIABELEN-SETUP
#
# Bouwt alle variabelen die de rest van je analyse nodig heeft:
#   epoch_data, estimates, hmp_features, times,
#   group_erps, subject_erps, difference, info, subject_epochs
#
# BELANGRIJKSTE FIX t.o.v. je oorspronkelijke twee losse scripts:
# Je HMP-script gebruikte per-SESSIE bestanden direct als "participant"
# in hmp.io.read_mne_data, terwijl je ERP-script sessies per SUBJECT
# samenvoegde. Daardoor hadden hmp_features en subject_erps/subject_epochs
# niet dezelfde participant-namen en epoch-nummering, en kon je ze niet
# betrouwbaar mergen voor de SVM/EEGNet-analyses.
#
# Oplossing hier: sessies worden EERST per subject samengevoegd
# (zoals in je ERP-script), opgeslagen als 1 bestand per subject, en
# dat samengevoegde bestand wordt vervolgens aan HMP gevoerd. Zo hebben
# epoch_data/estimates/hmp_features exact dezelfde participant-IDs en
# epoch-volgorde als subject_epochs/subject_erps.
# =====================================================================

import os
import re
import glob
import numpy as np
import pandas as pd
import mne
import hmp
from mne.io import read_info

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print(
        "pingouin niet gevonden -> ICC wordt later overgeslagen.\n"
        "Installeer met: pip install pingouin"
    )

# =====================================================================
# INSTELLINGEN -> pas aan naar jouw situatie
# =====================================================================
data_path = r"C:\Master Thesis BDS\data\preprocessed"
temp_path = os.path.join(data_path, "_subject_concat")  # voor samengevoegde per-subject bestanden
os.makedirs(temp_path, exist_ok=True)

sfreq = 256
conditions = ["target", "lure"]  # LET OP: moet overeenkomen met epochs.event_id keys


def to_canonical_id(filename):
    """Haalt 'sub-LTP063' uit bv. 'sub-LTP063_ses-1_epo.fif'."""
    match = re.match(r"(sub-[A-Za-z0-9]+)", filename)
    return match.group(1) if match else filename


all_files = [f for f in os.listdir(data_path) if f.endswith("_epo.fif")]
subjects = sorted(set(to_canonical_id(f) for f in all_files))
print(f"{len(subjects)} subjects gevonden:", subjects[:5])


# =====================================================================
# SECTION A: sessies per subject samenvoegen
#   -> subject_epochs (gebruikt voor ERP en later EEGNet)
#   -> 1 samengevoegd bestand per subject wegschrijven, voor HMP
# =====================================================================
subject_epochs = {}
subj_files = []   # paden naar samengevoegde per-subject bestanden (HMP input)
subj_names = []   # canonical subject namen, zelfde volgorde als subj_files

for sub in subjects:
    print("\n======================")
    print("Processing", sub)
    print("======================")

    files = glob.glob(os.path.join(data_path, f"{sub}_ses-*_epo.fif"))
    files = sorted(files, key=lambda x: int(x.split("_ses-")[1].split("_")[0]))
    print("Aantal sessies:", len(files))

    if len(files) == 0:
        print("Geen epochs gevonden, subject overgeslagen!")
        continue

    epochs_list = [mne.read_epochs(f, preload=True) for f in files]
    epochs_all = mne.concatenate_epochs(epochs_list)
    print("Aantal trials:", len(epochs_all))
    print("Events:", epochs_all.event_id)

    subject_epochs[sub] = epochs_all

    out_file = os.path.join(temp_path, f"{sub}_epo.fif")
    epochs_all.save(out_file, overwrite=True)
    subj_files.append(out_file)
    subj_names.append(sub)

print(f"\n{len(subj_files)} subjects klaar voor HMP-inlezen")
print(subj_files[:3])
print(subj_names[:3])


# =====================================================================
# SECTION B: channel informatie
# =====================================================================
info = read_info(subj_files[0], verbose=False)


# =====================================================================
# SECTION C: HMP dataset bouwen (nu met canonical, consistente namen)
# =====================================================================
epoch_data = hmp.io.read_mne_data(
    subj_files,
    data_format="epochs",
    sfreq=sfreq,
    verbose=False,
    subj_name=subj_names,
)
print(epoch_data)


# =====================================================================
# SECTION D: PCA preprocessing + HMP model fitten
# =====================================================================
preprocessed = hmp.preprocessors.ProjPCA(
    epoch_data,
    min_duration=.2,      # minimale reactietijd: 200ms
    max_duration=2,       # maximale reactietijd: 2s
    reject_threshold=1e-4,  # peak-to-peak rejection threshold (100 microV)
    interval_id="RT",
    n_comp=10,
)

model = hmp.models.CumulativeMethod()
_, estimates = model.fit_transform(preprocessed)


# =====================================================================
# SECTION E: single-trial HMP features (met epoch-index, voor latere merges)
# =====================================================================
def extract_hmp_features(estimates, n_events=None):
    """
    Single-trial HMP event latencies + probabilities per trial.
    'participant' is hier al canonical en 'epoch' komt overeen met de
    epoch-index in subject_epochs, dus dit is direct mergebaar met
    single-trial ERP features.
    """
    est = estimates.values
    n_trials = est.shape[0]
    if n_events is None:
        n_events = est.shape[2]

    rows = []
    for trial in range(n_trials):
        row = {
            "trial": trial,
            "condition": estimates.trial_type.values[trial],
            "participant": estimates.participant.values[trial],
            "epoch": estimates.epoch.values[trial],
            "RT": estimates.RT.values[trial],
            "correct": estimates.correct.values[trial],
        }
        for event in range(n_events):
            prob = est[trial, :, event]
            peak_sample = np.argmax(prob)
            latency_ms = (peak_sample / estimates.sfreq) * 1000
            row[f"event_{event + 1}_latency"] = latency_ms
            row[f"event_{event + 1}_prob"] = np.max(prob)
        rows.append(row)

    return pd.DataFrame(rows)


hmp_features = extract_hmp_features(estimates)
print(hmp_features.head())


# =====================================================================
# SECTION F: event times dataframe (zoals in je oorspronkelijke code)
# =====================================================================
times = hmp.utils.event_times(estimates, duration=True, add_rt=True)

first_channel = epoch_data.channel.values[0]

times = times.unstack().to_dataframe(name="duration")
times = times[~times.duration.isna()]  # verworpen trials eruit
times = times.reset_index().set_index(["participant", "epoch"])

times_metadata = (
    epoch_data
    .sel(sample=0, channel=first_channel)
    .to_dataframe()
    .iloc[:, 3:]
)
times_metadata = times_metadata.reset_index().set_index(["participant", "epoch"])
times = times.merge(times_metadata, on=["participant", "epoch"])
times = times.reset_index()


# =====================================================================
# SECTION G: ERP per subject + grand averages
#   (hergebruikt subject_epochs i.p.v. opnieuw in te laden)
# =====================================================================
subject_erps = {cond: [] for cond in conditions}

for sub, epochs_all in subject_epochs.items():
    for cond in conditions:
        if cond not in epochs_all.event_id:
            print(f"{sub}: conditie '{cond}' ontbreekt in event_id")
            continue
        epochs_cond = epochs_all[cond]
        print(sub, cond, "trials:", len(epochs_cond))
        subject_erps[cond].append(epochs_cond.average())

group_erps = {}
for cond in conditions:
    if len(subject_erps[cond]) == 0:
        print("Geen data voor", cond)
        continue
    group_erps[cond] = mne.grand_average(subject_erps[cond])

print("\nBeschikbare ERPs:", list(group_erps.keys()))

difference = mne.combine_evoked(
    [group_erps["target"], group_erps["lure"]],
    weights=[1, -1],
)
difference.comment = "Target - Lure"


# =====================================================================
# KLAAR -> beschikbare variabelen voor de rest van je analyse:
#   epoch_data, estimates, hmp_features, times,
#   group_erps, subject_erps, difference, info, subject_epochs
# =====================================================================
print("\nSetup compleet. Variabelen klaar:")
print("epoch_data, estimates, hmp_features, times, group_erps,")
print("subject_erps, difference, info, subject_epochs")

# # map waar je fif bestanden staan
# epoch_data_path = r"C:\Master Thesis BDS\data\preprocessed"


# # alle epochs files zoeken
# subj_files = [
#     os.path.join(epoch_data_path, f)
#     for f in os.listdir(epoch_data_path)
#     if f.endswith("_epo.fif")
# ]


# # subject namen halen
# subj_names = [
#     os.path.splitext(os.path.basename(f))[0]
#     for f in subj_files
# ]


# print(len(subj_files))
# print(subj_files[:3])
# print(subj_names[:3])


# # channel informatie
# info = read_info(
#     subj_files[0],
#     verbose=False
# )


# # sampling frequency
# sfreq = 256 

# # HMP dataset maken
# epoch_data = hmp.io.read_mne_data(
#     subj_files,
#     data_format="epochs",
#     sfreq=sfreq,
#     verbose=False,
#     subj_name=subj_names
# )

# print(epoch_data)

# # PCA preprocessing
# preprocessed = hmp.preprocessors.ProjPCA(epoch_data,
#                                         min_duration=.2, #Defining minimum reaction time as 200ms 
#                                         max_duration=2, # Maximum reaction time at 2s
#                                         reject_threshold=1e-4, #Peak-to-peak rejection threshold of 100 microV
#                                         interval_id = "RT", #In which variable of epoch_data are the duration to be modelled
#                                         n_comp=10, # How many principal components to keep for the fit, the more the better
#                                        )

# model = hmp.models.CumulativeMethod()

# _, estimates = model.fit_transform(
#     preprocessed
# )


def extract_erp_features(
    epochs,
    fn400_channel="E6",
    fn400_window=(0.300, 0.500),
    lpc_channel="E61",
    lpc_window=(0.500, 0.800),
    label_col="trial_type",
):
    """
    Berekent single-trial gemiddelde amplitude (FN400 en LPC) per epoch,
    voor 1 participant. Retourneert een DataFrame met 1 rij per trial.
    """
    fn400_data = epochs.copy().crop(*fn400_window).pick(fn400_channel).get_data()
    lpc_data = epochs.copy().crop(*lpc_window).pick(lpc_channel).get_data()
 
    fn400_amp = fn400_data.mean(axis=(1, 2)) * 1e6  # Volt -> microVolt
    lpc_amp = lpc_data.mean(axis=(1, 2)) * 1e6
 
    if epochs.metadata is not None and label_col in epochs.metadata.columns:
        condition = epochs.metadata[label_col].values
    else:
        # fallback: numerieke event-codes gebruiken
        condition = epochs.events[:, 2]
 
    df = pd.DataFrame(
        {
            "epoch": np.arange(len(epochs)),
            "condition": condition,
            "fn400_amp": fn400_amp,
            "lpc_amp": lpc_amp,
        }
    )
    return df
 
 
def build_subject_erp_features(subject_epochs, **kwargs):
    """
    Combineert single-trial ERP features over alle participants in
    subject_epochs. Extra kwargs (fn400_channel, lpc_window, label_col, ...)
    worden doorgegeven aan extract_erp_features.
    """
    all_feats = []
    for sub, epochs in subject_epochs.items():
        df = extract_erp_features(epochs, **kwargs)
        df["participant"] = sub
        all_feats.append(df)
    return pd.concat(all_feats, ignore_index=True)

# =====================================================================
# SECTION 1b: single-trial HMP features MET epoch-index (fix t.o.v. je code)
# =====================================================================
def extract_hmp_features(estimates, n_events=None):
    """
    Zelfde als je bestaande HMP single-trial loop, maar met 'epoch' erbij
    zodat je dit later kan mergen met de ERP features.
    """
    est = estimates.values
    n_trials = est.shape[0]
    if n_events is None:
        n_events = est.shape[2]

    rows = []
    for trial in range(n_trials):
        row = {
            "trial": trial,
            "condition": estimates.trial_type.values[trial],
            "participant": estimates.participant.values[trial],
            "epoch": estimates.epoch.values[trial],  # <-- toegevoegd
            "RT": estimates.RT.values[trial],
            "correct": estimates.correct.values[trial],
        }
        for event in range(n_events):
            prob = est[trial, :, event]
            peak_sample = np.argmax(prob)
            latency_ms = (peak_sample / estimates.sfreq) * 1000
            row[f"event_{event + 1}_latency"] = latency_ms
            row[f"event_{event + 1}_prob"] = np.max(prob)
        rows.append(row)

    return pd.DataFrame(rows)


# Gebruik:
# hmp_features = extract_hmp_features(estimates)


# =====================================================================
# SECTION 2: RELIABILITY -> split-half, Spearman-Brown, ICC
# =====================================================================
def split_half_reliability(
    df, value_col, participant_col="participant", n_iterations=100, random_state=None
):
    """
    Split-half betrouwbaarheid van een single-trial maat.

    Per participant: trials random in 2 helften splitsen, gemiddelde per
    helft berekenen, en de twee helften over participants heen correleren
    (Pearson r). Dit wordt n_iterations keer herhaald met random splits
    en gemiddeld, voor een stabiele schatting.
    """
    rng = np.random.default_rng(random_state)
    correlations = []

    for _ in range(n_iterations):
        half1_means, half2_means = {}, {}
        for participant, sub_df in df.groupby(participant_col):
            vals = sub_df[value_col].dropna().values
            if len(vals) < 4:
                continue
            idx = rng.permutation(len(vals))
            half1 = vals[idx[: len(vals) // 2]]
            half2 = vals[idx[len(vals) // 2 :]]
            half1_means[participant] = half1.mean()
            half2_means[participant] = half2.mean()

        common = sorted(set(half1_means) & set(half2_means))
        if len(common) < 3:
            continue
        h1 = np.array([half1_means[p] for p in common])
        h2 = np.array([half2_means[p] for p in common])
        r, _ = stats.pearsonr(h1, h2)
        correlations.append(r)

    mean_r = float(np.mean(correlations))
    spearman_brown = (2 * mean_r) / (1 + mean_r)

    return {
        "measure": value_col,
        "mean_split_half_r": mean_r,
        "spearman_brown_r": spearman_brown,
        "n_iterations": len(correlations),
    }


def odd_even_split_for_icc(df, value_col, participant_col="participant", epoch_col="epoch"):
    """
    Maakt een 'long format' DataFrame (1 rij per participant per helft)
    die je direct in pingouin.intraclass_corr kan stoppen.
    """
    rows = []
    for participant, sub_df in df.groupby(participant_col):
        sub_df = sub_df.sort_values(epoch_col)
        odd = sub_df.iloc[1::2][value_col].mean()
        even = sub_df.iloc[0::2][value_col].mean()
        rows.append({participant_col: participant, "half": "odd", value_col: odd})
        rows.append({participant_col: participant, "half": "even", value_col: even})
    return pd.DataFrame(rows)


def compute_icc(df, value_col, participant_col="participant"):
    """Berekent ICC(2,1) op basis van een odd/even split."""
    if not HAS_PINGOUIN:
        print("pingouin ontbreekt, ICC overgeslagen.")
        return None
    icc_df = odd_even_split_for_icc(df, value_col, participant_col)
    icc = pg.intraclass_corr(
        data=icc_df, targets=participant_col, raters="half", ratings=value_col
    )
    return icc


def run_reliability_analysis(feature_df):
    """
    Draait split-half + Spearman-Brown + ICC voor de belangrijkste maten:
    FN400 amplitude, LPC amplitude, en HMP event 2 / event 3 latency.
    feature_df = gemergde ERP + HMP single-trial dataframe (zie Section 3).
    """
    measures = ["fn400_amp", "lpc_amp", "event_2_latency", "event_3_latency"]
    reliability_results = []

    for measure in measures:
        if measure not in feature_df.columns:
            continue
        sh = split_half_reliability(feature_df, measure, n_iterations=200, random_state=0)
        reliability_results.append(sh)
        print(
            f"{measure}: split-half r = {sh['mean_split_half_r']:.3f}, "
            f"Spearman-Brown r = {sh['spearman_brown_r']:.3f}"
        )
        if HAS_PINGOUIN:
            icc = compute_icc(feature_df, measure)
            print(icc[["Type", "ICC", "CI95%"]])
            print()

    return pd.DataFrame(reliability_results)


# Gebruik:
# reliability_table = run_reliability_analysis(feature_df)


# =====================================================================
# SECTION 3: feature-dataframe samenvoegen (ERP + HMP)
# =====================================================================
def build_feature_dataframe(erp_features, hmp_features):
    """
    Merget single-trial ERP en HMP features op participant + epoch.
    Let op: pas de participant-namen indien nodig op elkaar aan
    (bv. 'sub-LTP063' vs 'sub-LTP063_epo' afhankelijk van subj_name).
    """
    feature_df = erp_features.merge(
        hmp_features, on=["participant", "epoch"], how="inner", suffixes=("", "_hmp")
    )
    return feature_df


# Gebruik:
# feature_df = build_feature_dataframe(erp_features, hmp_features)


# =====================================================================
# SECTION 4: NEUROPHYSIOLOGICAL PLAUSIBILITY -> statistische toetsen
# =====================================================================
def paired_condition_test(
    df,
    value_col,
    condition_col,
    cond_a,
    cond_b,
    participant_col="participant",
):
    """
    Paired t-test Target vs Lure op participant-gemiddelden van een maat
    (bv. FN400 amplitude, LPC amplitude, of HMP event duration).
    """
    means_a = df[df[condition_col] == cond_a].groupby(participant_col)[value_col].mean()
    means_b = df[df[condition_col] == cond_b].groupby(participant_col)[value_col].mean()
    common = means_a.index.intersection(means_b.index)

    t, p = stats.ttest_rel(means_a[common], means_b[common])
    d = (means_a[common].mean() - means_b[common].mean()) / means_a[common].sub(
        means_b[common]
    ).std()

    return {
        "measure": value_col,
        "t": t,
        "p": p,
        "n": len(common),
        "mean_target": means_a[common].mean(),
        "mean_lure": means_b[common].mean(),
        "cohens_d": d,
    }


def run_plausibility_tests(feature_df, times_df):
    """
    Voert Target-vs-Lure toetsen uit op:
      - FN400 amplitude (ERP)
      - LPC amplitude (ERP)
      - HMP event 2 duration (uit 'times' dataframe)
      - HMP event 3 duration (uit 'times' dataframe)
    """
    results = []

    results.append(
        paired_condition_test(
            feature_df, "fn400_amp", "condition", "RECOG_TARGET", "RECOG_LURE"
        )
    )
    results.append(
        paired_condition_test(
            feature_df, "lpc_amp", "condition", "RECOG_TARGET", "RECOG_LURE"
        )
    )

    for event in [2, 3]:
        sub = times_df[times_df["event"] == event]
        results.append(
            paired_condition_test(
                sub, "duration", "trial_type_x", "RECOG_TARGET", "RECOG_LURE"
            )
        )

    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df


# Gebruik:
# plausibility_table = run_plausibility_tests(feature_df, times)


# =====================================================================
# SECTION 5: FUNCTIONAL UTILITY -> SVM, within-subject CV, ROC, AUC
# =====================================================================
def within_subject_svm_auc(
    df,
    feature_cols,
    label_col="condition",
    participant_col="participant",
    n_splits=5,
    random_state=42,
):
    """
    Voor elke participant afzonderlijk: SVM-classificatie met stratified
    k-fold cross-validation BINNEN de participant (geen data-lekkage
    tussen subjects). Geeft AUC per participant + ROC-curve data terug.
    """
    results = []
    roc_curves = []

    for participant, sub_df in df.groupby(participant_col):
        X = sub_df[feature_cols].values
        y = sub_df[label_col].values

        if len(np.unique(y)) < 2 or len(y) < n_splits * 2:
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        y_true_all, y_score_all = [], []

        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            clf = SVC(kernel="rbf", probability=True, class_weight="balanced")
            clf.fit(X_train, y[train_idx])
            scores = clf.predict_proba(X_test)[:, 1]

            y_true_all.extend(y[test_idx])
            y_score_all.extend(scores)

        auc_score = roc_auc_score(y_true_all, y_score_all)
        fpr, tpr, _ = roc_curve(y_true_all, y_score_all)

        results.append({participant_col: participant, "auc": auc_score, "n_trials": len(y)})
        roc_curves.append((participant, fpr, tpr))

    return pd.DataFrame(results), roc_curves


def plot_mean_roc(roc_curves, title="ROC curve (within-subject)"):
    """Plot gemiddelde ROC curve +/- SD over participants."""
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for _, fpr, tpr in roc_curves:
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_fpr, mean_tpr, label=f"Mean ROC (AUC = {mean_auc:.3f})", lw=2)
    ax.fill_between(
        mean_fpr,
        np.clip(mean_tpr - std_tpr, 0, 1),
        np.clip(mean_tpr + std_tpr, 0, 1),
        alpha=0.2,
    )
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def compare_feature_sets(feature_df, feature_sets, label_col="condition", n_splits=5):
    """
    Vergelijkt classificatieprestatie (AUC) tussen featuresets,
    bv. {'ERP': [...], 'HMP': [...], 'ERP+HMP': [...], 'RIDE': [...]}.
    """
    all_results = []
    for name, cols in feature_sets.items():
        cols_present = [c for c in cols if c in feature_df.columns]
        if not cols_present:
            print(f"Geen kolommen gevonden voor featureset '{name}', overgeslagen.")
            continue
        res, _ = within_subject_svm_auc(
            feature_df, cols_present, label_col=label_col, n_splits=n_splits
        )
        res["feature_set"] = name
        all_results.append(res)

    comparison_df = pd.concat(all_results, ignore_index=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    comparison_df.boxplot(column="auc", by="feature_set", ax=ax)
    ax.set_ylabel("AUC (within-subject CV)")
    ax.set_title("Classificatieprestatie per featureset")
    plt.suptitle("")
    plt.tight_layout()

    print(comparison_df.groupby("feature_set")["auc"].agg(["mean", "std", "count"]))
    return comparison_df


# Gebruik:
# feature_sets = {
#     "ERP": ["fn400_amp", "lpc_amp"],
#     "HMP": [c for c in feature_df.columns if c.endswith("_latency")],
#     "ERP+HMP": ["fn400_amp", "lpc_amp"] + [c for c in feature_df.columns if c.endswith("_latency")],
#     # "RIDE": [...]  -> toevoegen zodra je RIDE-features hebt
# }
# comparison_df = compare_feature_sets(feature_df, feature_sets)


# =====================================================================
# SECTION 6: PSEUDO-TRIAL ANALYSE -> AUC als functie van SNR
# (uitbreiding n.a.v. feedback begeleider)
# =====================================================================
def make_pseudo_trials(
    df,
    feature_cols,
    label_col="condition",
    participant_col="participant",
    n_average=4,
    n_repeats=20,
    random_state=None,
):
    """
    Maakt pseudo-trials door n_average single trials (zelfde participant +
    conditie) willekeurig te middelen. Dit verhoogt de SNR van de features
    en simuleert wat er gebeurt als je over meerdere trials middelt
    vóór classificatie.
    """
    rng = np.random.default_rng(random_state)
    rows = []

    for (participant, cond), sub_df in df.groupby([participant_col, label_col]):
        X = sub_df[feature_cols].values
        n_trials = len(X)
        if n_trials < n_average:
            continue
        for _ in range(n_repeats):
            idx = rng.choice(n_trials, size=n_average, replace=False)
            pseudo = X[idx].mean(axis=0)
            row = dict(zip(feature_cols, pseudo))
            row[participant_col] = participant
            row[label_col] = cond
            rows.append(row)

    return pd.DataFrame(rows)


def pseudo_trial_snr_curve(
    df,
    feature_cols,
    label_col="condition",
    participant_col="participant",
    n_levels=(1, 2, 4, 8, 16),
    n_repeats=20,
    random_state=42,
):
    """
    Herhaalt de within-subject SVM-classificatie voor verschillende
    aantallen gemiddelde trials per pseudo-trial (1 = single-trial).
    Geeft een DataFrame terug met AUC per participant per niveau,
    zodat je de 'SNR-plafond' curve kan plotten.
    """
    snr_results = []

    for n in n_levels:
        if n == 1:
            level_df = df.copy()
            n_splits = 5
        else:
            level_df = make_pseudo_trials(
                df,
                feature_cols,
                label_col=label_col,
                participant_col=participant_col,
                n_average=n,
                n_repeats=n_repeats,
                random_state=random_state,
            )
            # minder folds bij minder pseudo-trials per participant/conditie
            n_splits = 3

        res, _ = within_subject_svm_auc(
            level_df,
            feature_cols,
            label_col=label_col,
            participant_col=participant_col,
            n_splits=n_splits,
            random_state=random_state,
        )
        res["n_averaged"] = n
        snr_results.append(res)
        print(f"n_averaged = {n}: gemiddelde AUC = {res['auc'].mean():.3f}")

    return pd.concat(snr_results, ignore_index=True)


def plot_snr_curve(snr_df, title="AUC als functie van trial-averaging (SNR)"):
    summary = snr_df.groupby("n_averaged")["auc"].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        summary.index,
        summary["mean"],
        yerr=summary["std"],
        marker="o",
        capsize=4,
        lw=2,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(summary.index)
    ax.set_xticklabels(summary.index)
    ax.set_xlabel("Aantal gemiddelde trials per pseudo-trial")
    ax.set_ylabel("AUC (within-subject CV)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


# Gebruik:
# snr_df = pseudo_trial_snr_curve(
#     feature_df,
#     feature_cols=["fn400_amp", "lpc_amp"],   # of een andere featureset
#     n_levels=(1, 2, 4, 8, 16),
# )
# plot_snr_curve(snr_df)


# =====================================================================
# SECTION 7: VOORBEELD VAN VOLLEDIGE PIPELINE (pas aan naar jouw situatie)
# =====================================================================
if __name__ == "__main__":
    # Dit blok draait alleen als je dit bestand los uitvoert, en gaat ervan
    # uit dat subject_epochs, estimates, times etc. al gegenereerd zijn
    # door je bestaande scripts (zoals hierboven beschreven in Section 0).

    # 1) Features bouwen
    erp_features = build_subject_erp_features(subject_epochs)
    hmp_features = extract_hmp_features(estimates)
    feature_df = build_feature_dataframe(erp_features, hmp_features)

    # 2) Reliability
    reliability_table = run_reliability_analysis(feature_df)

    # 3) Neurophysiological plausibility
    plausibility_table = run_plausibility_tests(feature_df, times)

    # 4) Functional utility
    feature_sets = {
        "ERP": ["fn400_amp", "lpc_amp"],
        "HMP": [c for c in feature_df.columns if c.endswith("_latency")],
        "ERP+HMP": ["fn400_amp", "lpc_amp"]
        + [c for c in feature_df.columns if c.endswith("_latency")],
    }
    comparison_df = compare_feature_sets(feature_df, feature_sets)

    # 5) Pseudo-trial / SNR-analyse
    snr_df = pseudo_trial_snr_curve(
        feature_df, feature_cols=["fn400_amp", "lpc_amp"], n_levels=(1, 2, 4, 8, 16)
    )
    plot_snr_curve(snr_df)

    plt.show()