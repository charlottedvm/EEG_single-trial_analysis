def extract_features_hmp(data)

    """
    data:
    trials x channels x time
    """


    model = hmp.fit(
        data
    )


    stages = model.get_states()


    features = {

        "stage1_latency":
            stages[0]["onset"],

        "stage2_latency":
            stages[1]["onset"],

        "stage3_latency":
            stages[2]["onset"],


        "stage1_amplitude":
            stages[0]["amplitude"],

        "stage2_amplitude":
            stages[1]["amplitude"],

        "stage3_amplitude":
            stages[2]["amplitude"]

    }


    return features