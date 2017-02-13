from collections import Counter

import numpy
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight


def get_class_labels(y):
    """
    Get the class labels
    :param y: list of labels, ex. ['positive', 'negative', 'positive', 'neutral', 'positive', ...]
    :return: sorted unique class labels
    """
    return numpy.unique(y)


def labels_to_categories(y):
    """
    Labels to categories
    :param y: list of labels, ex. ['positive', 'negative', 'positive', 'neutral', 'positive', ...]
    :return: list of categories, ex. [0, 2, 1, 2, 0, ...]
    """
    encoder = LabelEncoder()
    encoder.fit(y)
    y_num = encoder.transform(y)
    return y_num


def get_labels_to_categories_map(y):
    """
    Get the mapping of class labels to numerical categories
    :param y: list of labels, ex. ['positive', 'negative', 'positive', 'neutral', 'positive', ...]
    :return: dictionary with the mapping
    """
    labels = get_class_labels(y)
    return {l: i for i, l in enumerate(labels)}


def categories_to_onehot(y):
    """
    Transform categorical labels to one-hot vectors
    :param y: list of categories, ex. [0, 2, 1, 2, 0, ...]
    :return: list of one-hot vectors, ex. [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], ...]
    """
    return np_utils.to_categorical(y)


def onehot_to_categories(y):
    """
    Transform categorical labels to one-hot vectors
    :param y: list of one-hot vectors, ex. [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], ...]
    :return: list of categories, ex. [0, 2, 1, 2, 0, ...]
    """
    return numpy.asarray(y).argmax(axis=-1)


def get_class_weights(y):
    """
    Returns the normalized weights for each class based on the frequencies of the samples
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """

    weights = compute_class_weight('balanced', numpy.unique(y), y)

    d = {c: w for c, w in zip(numpy.unique(y), weights)}

    return d


def get_class_weights2(y, smooth_factor=0):
    """
    Returns the normalized weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}


def print_dataset_statistics(y):
    """
    Returns the normalized weights for each class based on the frequencies of the samples
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)
    print("Total:", len(y))
    statistics = {c: str(counter[c]) + " (%.2f%%)" % (counter[c] / float(len(y)) * 100.0)
                  for c in sorted(counter.keys())}
    print(statistics)


def predic_classes(pred):
    if pred.shape[-1] > 1:
        return pred.argmax(axis=-1)
    else:
        return (pred > 0.5).astype('int32')
