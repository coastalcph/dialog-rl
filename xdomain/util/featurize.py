import numpy as np
from collections import OrderedDict
import torch


def make_n_gram_bow(sequence, n, mode='sum', vectors=True):
    """
    Aggregates over sliding window of `n' concatenations of word vectors.
    Vectors can either be fixed-size embeddings or one-hot vectors. In the
    latter case, it's a straight BOW model, no concatenation of n-grams for
    now (should this be implemented? seems like this would grow the vectors
    really large as they get length V**n).
    :param sequence:
    :param n:
    :param mode:
    :param vectors:
    :return:
    """
    bow = []
    assert len(sequence) > n, "Sequence too short (must be at least n+1 long)"
    if vectors:
        for i in range(len(sequence) - n + 1):
            bow.append(np.concatenate(sequence[i:i+n]))
    else:
        #  TODO actually do n-gram BOW (every n-gram gets its index...)
        bow = sequence  # just aggregate over one-hot vectors
    if mode == 'sum':
        return np.sum(bow, 0)
    elif mode == 'avg':
        return np.mean(bow, 0)
    elif mode == 'max':
        return np.max(bow, 0)
    else:
        return NotImplementedError("Mode for aggregating n-grams must be one "
                                   "of 'sum', 'avg' or 'max'.")


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

    def __init__(self, embeddings, n=2):
        """

        :param embeddings: Embeddings dictionary, mapping words to fix length
        vectors
        :param n: the order for n-gram concatenations (see StateNet paper)
        """
        super().__init__()
        self.embeddings = embeddings
        self.n = n

    def featurize_word(self, word):
        return np.array(self.embeddings.get(word.lower()),
                        dtype=np.float)

    def featurize_turn(self, turn):
        if type(turn) == str:
            turn = turn.split()
        turn = ['<sos>'] + turn + ['<eos>']
        # if not turn:
        #     return torch.zeros(len(self.embeddings['i']) * self.n)
        seq = [self.featurize_word(w) for w in turn if w in self.embeddings]
        if len(seq) < (self.n+1):
            seq += [self.featurize_word("<eos>") for _ in range(self.n+1 -
                                                                len(seq))]
        ngrams = make_n_gram_bow(seq, self.n, mode='sum')
        # print(seq, ngrams, ngrams.shape)
        # print(turn )
        return torch.Tensor(ngrams)

    def featurize_dialog(self, dialog):
        return [self.featurize_turn(t) for t in dialog.to_dict()['turns']]


class ActionFeaturizer(Featurizer):

    def __init__(self, embeddings, mode='max'):
        """

        :param embeddings: Embeddings dictionary, mapping words to fix length
        vectors
        """
        super().__init__()
        self.embeddings = embeddings
        self.mode = mode

    def featurize_act(self, act):
        if self.mode == 'max':
            # maxpooling over words in act (['inform', 'Price', '=', 'cheap'])
            vec = torch.Tensor(np.max(
                [np.array(self.embeddings.get(w.lower()), dtype=np.float)
                 for w in act if w.lower() in self.embeddings], axis=0))
        else:
            raise NotImplementedError('only max pooling implemented so far')
        return vec

    def featurize_turn(self, turn):
        if turn:
            # print(turn)
            act_f = torch.stack([self.featurize_act(a) for a in turn])
            # print(act_f)
            return torch.max(act_f, 0)[0]
            # return np.array(act_f).sum()

        else:
            return torch.zeros(len(self.embeddings['i']))


class SlotFeaturizer(Featurizer):

    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def featurize_slot(self, slot):
        e = self.embeddings.get(slot)
        if not e:
            e = np.zeros(len(self.embeddings.get("i")))
        return torch.Tensor(e)


class ValueFeaturizer(Featurizer):

    def __init__(self):
        super().__init__()

