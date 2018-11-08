import numpy as np
from collections import OrderedDict


class LabelMapper:
    """
    Encodes true values as one-hot vectors
    """
    def __init__(self):
        super().__init__()
        self.label_values = set()
        self.label2id = OrderedDict()
        self.id2label = OrderedDict()

    def fit(self, labels, warm_start=False):
        if not warm_start:
            self.label_values = set()
        for label in labels:
            self.label_values.add(label)
        _values = sorted(list(self.label_values))
        for i, v in enumerate(_values):
            self.label2id[v] = i
            self.id2label[i] = v

    def transform(self, labels, onehot=True):
        """

        :param labels:
        :param onehot: If true, encode labels to onehot. Else, encode to
        categorical representations (integers)
        :return:
        """
        out = []
        for l in labels:
            if onehot:
                encoding = np.zeros(len(self.label_values))
                encoding[self.label2id[l]] = 1
            else:
                encoding = self.label2id[l]
            out.append(encoding)
        return out

    def fit_transform(self, labels, onehot=True):
        self.fit(labels)
        return self.transform(labels, onehot)


class Featurizer:

    def __init__(self):
        pass

    def fit_transform(self, inputs):
        pass

    def fit(self, inputs):
        pass


class UserInputFeaturizer(Featurizer):

    def __init__(self):
        super().__init__()


class ActionFeaturizer(Featurizer):

    def __init__(self):
        super().__init__()


class SlotFeaturizer(Featurizer):

    def __init__(self):
        super().__init__()


class ValueFeaturizer(Featurizer):

    def __init__(self):
        super().__init__()

