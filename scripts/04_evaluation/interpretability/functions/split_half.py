import numpy as np


def split_half_stability(
        trials,
        feature_function,
        n_iterations=1000,
        random_state=42
):
    """
    Generic within-subject split-half stability.

    trials:
        ndarray (n_trials, channels, times)

    feature_function:
        functie die trials neemt en features teruggeeft als dict

    Bijvoorbeeld:
        {
        "latency": 250,
        "amplitude": 5.2
        }

    """

    rng = np.random.default_rng(random_state)

    differences = {}


    for _ in range(n_iterations):

        # random trial split
        indices = rng.permutation(len(trials))

        half = len(indices) // 2

        idx1 = indices[:half]
        idx2 = indices[half:]


        trials_1 = trials[idx1]
        trials_2 = trials[idx2]


        # RIDE of HMP hier
        features_1 = feature_function(trials_1)
        features_2 = feature_function(trials_2)


        for feature in features_1.keys():

            if feature not in differences:
                differences[feature] = []


            diff = abs(
                features_1[feature]
                -
                features_2[feature]
            )

            differences[feature].append(diff)



    results = {}

    for feature, values in differences.items():

        values = np.array(values)

        results[feature] = {

            "mean_absolute_difference":
                np.mean(values),

            "sd":
                np.std(values),

            "ci95":
                np.percentile(
                    values,
                    [2.5,97.5]
                )
        }


    return results