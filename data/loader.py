import numpy as np
import copy


def data_process(indexes, times, attributes, values, results=None, max_input_length=200):
    label = []
    data = []
    mask = []
    template = np.transpose([indexes, np.zeros_like(indexes)])
    padding = np.zeros([max_input_length-len(indexes), 2])
    template = np.concatenate((template, padding), 0)
    for i in times:
        for j in attributes:
            for k in values:
                data_temp = copy.deepcopy(template)
                for g in range(max_input_length):
                    data_temp[g][1] = pow(0.5, min_dis(g, [i, j, k]) + 1) * 256
                data_temp[i][1] = 256  # Time  label 300
                data_temp[j][1] = 256  # Attr  label 300
                data_temp[k][1] = 256  # Value label 300
                data.append(data_temp)
                mask.append([i, j, k])
                label.append(1 if [i, j, k] in results else 0)
    mask = np.int32(mask)
    data = np.float32(data)
    label = np.float32(label)
    # mask = list(range(len(indexes)))
    return data, label, mask


def min_dis(index, array):
    min = np.abs(index - array[0])
    for i in range(1, len(array)):
        if np.abs(index - array[i]) < min:
            min = np.abs(index - array[i])
    return min
