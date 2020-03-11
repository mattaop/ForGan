import numpy as np


# split a univariate sequence into samples
def split_sequence(sequence, sequence_length, forecast_hoizon, num_features=1):
    x = np.zeros([len(sequence)-sequence_length-forecast_hoizon + 1, sequence_length, 1])
    y = np.zeros([len(sequence)-sequence_length-forecast_hoizon + 1, forecast_hoizon, 1])
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + sequence_length
        out_end_ix = end_ix + forecast_hoizon
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        y[i] = seq_y
        x[i] = seq_x
    return x, y
