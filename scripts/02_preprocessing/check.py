import mne
import glob
import os
import matplotlib.pyplot as plt


# =====================================================
# Instellingen
# =====================================================

data_path = r"C:\Master Thesis BDS\data\preprocessed"


subjects = [
    "sub-LTP063",
    "sub-LTP065",
    # voeg hier meer subjects toe
]


conditions = [
    "target",
    "lure"
]


# =====================================================
# Opslag participant ERPs
# =====================================================

subject_erps = {
    cond: []
    for cond in conditions
}



# =====================================================
# Loop over subjects
# =====================================================

for sub in subjects:

    print("\n======================")
    print("Processing", sub)
    print("======================")


    # zoek alle sessies
    files = glob.glob(
        os.path.join(
            data_path,
            f"{sub}_ses-*_epo.fif"
        )
    )


    # numeriek sorteren
    files = sorted(
        files,
        key=lambda x: int(
            x.split("_ses-")[1].split("_")[0]
        )
    )


    print("Aantal sessies:", len(files))


    if len(files) == 0:

        print("Geen epochs gevonden!")
        continue



    epochs_list = []


    # =================================================
    # laad sessies
    # =================================================

    for f in files:

        print("Loading:", os.path.basename(f))


        epochs = mne.read_epochs(
            f,
            preload=True
        )


        epochs_list.append(
            epochs
        )



    # =================================================
    # combineer sessies participant
    # =================================================

    epochs_all = mne.concatenate_epochs(
        epochs_list
    )


    print(
        "Aantal trials:",
        len(epochs_all)
    )


    print(
        "Events:",
        epochs_all.event_id
    )



    # =================================================
    # ERP per conditie
    # =================================================

    for cond in conditions:


        if cond not in epochs_all.event_id:

            print(
                f"{cond} ontbreekt"
            )

            continue



        epochs_cond = epochs_all[cond]


        print(
            sub,
            cond,
            "trials:",
            len(epochs_cond)
        )


        erp = epochs_cond.average()


        subject_erps[cond].append(
            erp
        )



# =====================================================
# Grand average over subjects
# =====================================================

group_erps = {}


for cond in conditions:


    if len(subject_erps[cond]) == 0:

        print(
            "Geen data voor",
            cond
        )

        continue



    print(
        "Grand average:",
        cond
    )


    group_erps[cond] = mne.grand_average(
        subject_erps[cond]
    )



print("\nBeschikbare ERPs:")
print(group_erps.keys())



# =====================================================
# Controle channels
# =====================================================

print("\nChannels:")
print(
    group_erps["target"].ch_names
)



# =====================================================
# Difference wave
# Target - Lure
# =====================================================

difference = mne.combine_evoked(
    [
        group_erps["target"],
        group_erps["lure"]
    ],
    weights=[
        1,
        -1
    ]
)


difference.comment = "Target - Lure"



# =====================================================
# FN400
# E6
# 300-500 ms
# =====================================================


fig, ax = plt.subplots(
    figsize=(8,4)
)


mne.viz.plot_compare_evokeds(
    {
        "Target": group_erps["target"],
        "Lure": group_erps["lure"]
    },
    picks="E6",
    axes=ax,
    ci=False,
    show=False
)


ax.axvspan(
    0.300,
    0.500,
    alpha=0.3
)


ax.set_title(
    "FN400 - E6"
)


ax.set_xlabel(
    "Time (s)"
)


ax.set_ylabel(
    "Amplitude (µV)"
)


plt.tight_layout()



# =====================================================
# LPC
# E61
# 500-800 ms
# =====================================================


fig, ax = plt.subplots(
    figsize=(8,4)
)


mne.viz.plot_compare_evokeds(
    {
        "Target": group_erps["target"],
        "Lure": group_erps["lure"]
    },
    picks="E61",
    axes=ax,
    ci=False,
    show=False
)


ax.axvspan(
    0.500,
    0.800,
    alpha=0.3
)


ax.set_title(
    "LPC - E61"
)


ax.set_xlabel(
    "Time (s)"
)


ax.set_ylabel(
    "Amplitude (µV)"
)


plt.tight_layout()



# =====================================================
# Difference wave plot
# =====================================================


mne.viz.plot_compare_evokeds(
    {
        "Target - Lure": difference
    },
    picks=[
        "E6",
        "E61"
    ]
)



# =====================================================
# Mean amplitudes
# =====================================================


fn400 = (
    difference
    .copy()
    .crop(
        0.300,
        0.500
    )
    .pick(
        "E6"
    )
    .data
    .mean()
)


lpc = (
    difference
    .copy()
    .crop(
        0.500,
        0.800
    )
    .pick(
        "E61"
    )
    .data
    .mean()
)



print(
    "\nFN400 Target-Lure:",
    round(fn400,3),
    "µV"
)


print(
    "LPC Target-Lure:",
    round(lpc,3),
    "µV"
)



# =====================================================
# Topografie
# =====================================================


difference.plot_topomap(
    times=[0.400],
    average=0.100,
    title="FN400 topography (400 ms)"
)



difference.plot_topomap(
    times=[0.650],
    average=0.150,
    title="LPC topography (650 ms)"
)



plt.show()