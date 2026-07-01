def extract_features_ride(trials):

    result = ride(
        trials
    )

    return {

        "C_latency": result.C_latency,

        "C_amplitude": result.C_amplitude,

        "S_latency": result.S_latency,

        "S_amplitude": result.S_amplitude

    }