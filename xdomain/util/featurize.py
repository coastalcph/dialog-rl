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

    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def featurize_word(self, word):
        return np.array(self.embeddings.get(word), dtype=np.float32)

    def featurize_turn(self, turn):
        return [self.featurize_word(w) for w in turn if w in self.embeddings]

    def featurize_dialog(self, dialog):
        return [self.featurize_turn(t) for t in dialog.to_dict()['turns']]


class ActionFeaturizer(Featurizer):

    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def featurize_act(self, act):
        vec = np.mean([np.array(self.embeddings.get(w), dtype=np.float32)
                       for w in act if w in self.embeddings], axis=0)
        print(vec, vec.shape)
        return vec

    def featurize_turn(self, turn):
        print(turn)
        return [self.featurize_act(a) for a in turn]

class SlotFeaturizer(Featurizer):

    def __init__(self):
        super().__init__()


class ValueFeaturizer(Featurizer):

    def __init__(self):
        super().__init__()

