# =============================================================================
# config.py — Centrale instellingen voor de PEERS EEG pipeline
# =============================================================================

from pathlib import Path

# ---------------------------
# Paden
# ---------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "preprocessed"
DERIVATIVES_DIR = PROJECT_DIR / "data" / "derivatives"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)

def raw_path(sub, ses):
    return str(RAW_DIR / sub / ses / 'eeg' / f'{sub}_{ses}_task-ltpFR_eeg.edf')

def events_path(sub, ses):
    return str(RAW_DIR / sub / ses / 'eeg' / f'{sub}_{ses}_task-ltpFR_events.tsv')

def json_path(sub, ses):
    return str(RAW_DIR / sub / ses / 'eeg' / f'{sub}_{ses}_task-ltpFR_eeg.json')

def out_file(sub, ses):
    return OUTPUT_DIR / f'{sub}_{ses}_epo.fif'

def derivatives_file(sub, ses):
    return DERIVATIVES_DIR / f'{sub}_{ses}_derivatives.pkl'

# ---------------------------
# Subjects & sessions
# ---------------------------
def get_subjects() -> list[str]:
    if not RAW_DIR.exists():
        return []
    return sorted([
        d.name for d in RAW_DIR.iterdir()
        if d.is_dir() and d.name.startswith('sub-')
    ])

def get_sessions(sub: str) -> list[str]:
    sub_dir = RAW_DIR / sub
    if not sub_dir.exists():
        return []
    return sorted([
        d.name for d in sub_dir.iterdir()
        if d.is_dir() and d.name.startswith('ses-')
    ])

# ---------------------------
# Kanalen
# ---------------------------
FACE_CHANNELS = [
    'E1', 'E2', 'E3', 'E9', 'E10', 'E14', 'E15', 'E16', 'E17', 'E18',
    'E21', 'E22', 'E23', 'E26', 'E27', 'E32', 'E33', 'E116', 'E122',
    'E123', 'E125', 'E128', 'E8', 'E25', 'E126', 'E127'
]
MASTOIDS           = ['E57', 'E100']
PROTECTED_CHANNELS = FACE_CHANNELS + MASTOIDS + ['Cz']
MONTAGE_NAME       = 'GSN-HydroCel-129'

# ---------------------------
# Filter & signaal
# ---------------------------
LOW_PASS   = 30     # Hz
HIGH_PASS  = 0.1    # Hz
NOTCH_FREQ = 50     # Hz (Europees net)
RESAMPLE   = 256    # Hz

# ---------------------------
# Epochs
# ---------------------------
EPOCH_TMIN = -1.0   # sec
EPOCH_TMAX =  4.0   # sec
BASELINE   = (-0.2, 0)

# ---------------------------
# Gedrag
# ---------------------------
RT_MIN          = 0.3   # sec
RT_MAX          = 2.0   # sec
MIN_PERFORMANCE = 0.55
MIN_TRIALS      = 20

# ---------------------------
# Bad channel drempels
# ---------------------------
BAD_CH_SD_FACTOR       = 5      # drift: kanaalgemiddelde > N × SD
BAD_CH_AMP_THRESH      = 500e-6 # V — amplitude drempel
BAD_CH_AMP_FRAC        = 0.2   # fractie tijdstappen boven drempel
BAD_CH_VAR_SD_FACTOR   = 3      # variantie: kanaal-SD > N × SD
BAD_CH_EPOCH_THRESHOLD = 0.10   # epoch-fractie voordat kanaal geïnterpoleerd wordt

# ---------------------------
# ICA
# ---------------------------
ICA_N_COMPONENTS  = 0.99
ICA_METHOD        = 'infomax'
ICA_RANDOM_STATE  = 42
ICA_MAX_ITER      = 1024
ICA_EYE_PROB      = 0.75    # minimale kans om als oogcomponent te tellen
ICA_HIGH_PASS     = 1.0     # Hz — apart gefilterd voor ICLabel
ICA_LOW_PASS      = 100.0   # Hz

# ---------------------------
# Epoch rejection
# ---------------------------
REJECT_FIRST  = dict(eeg=500e-6)  # V — eerste rejectie (voor ICA)
REJECT_FINAL  = dict(eeg=200e-6)  # V — finale rejectie (na ICA + baseline)

# ---------------------------
# Overig
# ---------------------------
FORCE_REPROCESS = False
EVENT_DICT      = {'RECOG_TARGET': 1, 'RECOG_LURE': 2}