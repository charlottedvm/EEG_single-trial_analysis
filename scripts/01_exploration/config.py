# =============================================================================
# config.py — Shared constants for the PEERS dataset pipeline
# =============================================================================

# Subjects & sessions
SUBJECTS = ['sub-LTP063', 'sub-LTP064', 'sub-LTP065']
SESSIONS = [f'ses-{i}' for i in range(20)]

# ---------------------------
RAW_DIR    = Path('./data/raw')
OUTPUT_DIR = Path('./data/preprocessed')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def raw_path(sub, ses):
    return str(RAW_DIR / sub / ses / 'eeg' / f'{sub}_{ses}_task-ltpFR_eeg.edf')

def events_path(sub, ses):
    return str(RAW_DIR / sub / ses / 'eeg' / f'{sub}_{ses}_task-ltpFR_events.tsv')

def json_path(sub, ses):
    return str(RAW_DIR / sub / ses / 'eeg' / f'{sub}_{ses}_task-ltpFR_eeg.json')

def out_file(sub, ses):
    return OUTPUT_DIR / f'{sub}_{ses}_epo.fif'

# Electrode layout
FACE_CHANNELS = [
    'E1',  'E2',  'E3',  'E9',  'E10', 'E14', 'E15', 'E16', 'E17', 'E18',
    'E21', 'E22', 'E23', 'E26', 'E27', 'E32', 'E33', 'E116','E122',
    'E123','E125','E128','E8',  'E25', 'E126','E127'
]
VEOG_CHANNELS = ['E8', 'E25']
HEOG_CHANNELS = ['E126', 'E127']
EOG_CHANNELS  = VEOG_CHANNELS + HEOG_CHANNELS
MASTOIDS      = ['E57', 'E100']
MONTAGE_NAME  = 'GSN-HydroCel-129'

# Coordinate thresholds for face channel detection
Z_THRESH = 0.08  # z < this → face/chin area
Y_THRESH = 0.047  # y > this → front of head

# Behavioral filtering
RT_MIN = 0.3   # seconds
RT_MAX = 2.0   # seconds

# Bad channel thresholds
BAD_CH_MEAN_SD_FACTOR = 5    # channel mean > N * SD of all channel means
BAD_CH_AMP_THRESH_UV  = 500  # µV — channel flagged if >20% of samples exceed this
BAD_CH_VAR_SD_FACTOR  = 3    # channel SD > N * SD of all channel SDs

# Data path template  (use .format(sub=..., ses=...))
EEG_PATH    = 'data/{sub}/{ses}/eeg/{sub}_{ses}_task-ltpFR_eeg.edf'
EVENTS_PATH = 'data/{sub}/{ses}/eeg/{sub}_{ses}_task-ltpFR_events.tsv'
JSON_PATH   = 'data/{sub}/{ses}/eeg/{sub}_{ses}_task-ltpFR_eeg.json'
ELEC_PATH   = 'data/{sub}/{ses}/eeg/{sub}_{ses}_space-CapTrak_electrodes.tsv'
CHAN_PATH   = 'data/{sub}/{ses}/eeg/{sub}_{ses}_task-ltpFR_channels.tsv'
BEH_PATH    = 'data/{sub}/{ses}/beh/{sub}_{ses}_task-ltpFR_beh.json'