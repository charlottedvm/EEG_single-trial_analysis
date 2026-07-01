import numpy as np


# gemiddelde tijd waarop elk event maximaal waarschijnlijk is
event_latencies = []


for trial in range(estimates.shape[0]):

    trial_events = []

    for event in range(estimates.shape[2]):

        peak_sample = np.argmax(
            estimates[trial,:,event]
        )


        latency = (
            preprocessed.times[peak_sample]
            * 1000
        )


        trial_events.append(
            latency
        )


    event_latencies.append(
        trial_events
    )


event_latencies = np.array(event_latencies)


print(event_latencies.shape)