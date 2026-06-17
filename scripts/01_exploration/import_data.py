import openneuro as on

subjects = [
    "sub-LTP063",
    # "sub-LTP064",
    # "sub-LTP065"
]

for sub in subjects:
    on.download(
        dataset="ds004395",
        target_dir="data/raw",
        include=[sub]
    )