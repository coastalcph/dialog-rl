import numpy as np
from collections import OrderedDict
import torch
from util.data import *
from tqdm import tqdm


def get_value_index(_values, _val):
    for _idx, candidate in enumerate(_values):
        if candidate.value == _val:
            return _idx
    return -1


def featurize_dialogs_elmo(_data, s2v, device, args, pooled=True):
    featurized_dialogs = []

    for dg in tqdm(_data):
        featurized_turns = []

        all_user_utts = []
        all_system_acts = []
        all_system_utts = []
        all_lbls = []
        all_ys = []
        all_bsts = []

        for t in dg['turns']:
            if args.pooled:
                utt = t['usr_trans_elmo_pool']
                sys = t['sys_trans_elmo_pool']
                act = t['sys_acts_elmo_pool']
            else:
                utt = t['usr_trans_elmo']
                sys = t['sys_trans_elmo']
                act = t['sys_acts_elmo']
            bst = t['belief_state']
            lbls = {}
            for s, v in t['turn_label']:
                v = v.lower()
                if v in [_v.value for _v in s2v[s].values]:
                    lbls[s] = v
                else:
                    lbls[s] = "<true>"
            ys = {}
            for slot, val in lbls.items():
                values = s2v[slot].values
                ys[slot] = torch.zeros(len(values))
                idx = get_value_index(values, val)
                ys[slot][idx] = 1

            all_user_utts.append(utt)
            all_system_acts.append(act)
            all_system_utts.append(sys)
            all_ys.append(ys)
            all_lbls.append(lbls)
            all_bsts.append(bst)

        all_x_utt = all_user_utts
        all_x_act = all_system_acts
        all_x_sys = all_system_utts

        for i in range(len(dg['turns'])):

            # Encode user and action representations
            if args.pooled:
                x_utt = all_x_utt[i].to(device)
                x_sys = all_x_sys[i].to(device)
                x_act = all_x_act[i].to(device)
            else:
                x_utt = [t.to(device) for t in
                         all_x_utt[i]]  # one vector per n
                x_sys = [t.to(device) for t in
                         all_x_sys[i]]  # one vector per n
                x_act = [t.to(device) for t in
                         all_x_act[i]]

            featurized_turns.append(Turn(
                all_user_utts[i], all_system_acts[i], all_system_utts[i],
                x_utt, x_act, x_sys,
                all_ys[i], all_lbls[i], all_bsts[i]))

        featurized_dialogs.append(Dialog(featurized_turns))

    return featurized_dialogs


def featurize_dialogs(_data, s2v, device, args, w2v=None):
    featurized_dialogs = []

    utt_ftz = UserInputNgramFeaturizer(w2v, n=args.M)
    sys_ftz = UserInputNgramFeaturizer(w2v, n=args.M)
    act_ftz = ActionFeaturizer(w2v)

    for dg in tqdm(_data):
        featurized_turns = []

        all_user_utts = []
        all_system_acts = []
        all_system_utts = []
        all_lbls = []
        all_ys = []
        all_bsts = []

        for t in dg['turns']:
            utt = t['transcript']
            sys = t['system_transcript']
            act = t['system_acts']
            bst = t['belief_state']
            lbls = {}
            for s, v in t['turn_label']:
                v = v.lower()
                if v in [_v.value for _v in s2v[s].values]:
                    lbls[s] = v
                else:
                    lbls[s] = "<true>"

            ys = {}
            for slot, val in lbls.items():
                values = s2v[slot].values
                ys[slot] = torch.zeros(len(values))
                idx = get_value_index(values, val)
                ys[slot][idx] = 1

            all_user_utts.append(utt)
            all_system_acts.append(act)
            all_system_utts.append(sys)
            all_ys.append(ys)
            all_lbls.append(lbls)
            all_bsts.append(bst)

        all_x_utt = utt_ftz.featurize_batch(all_user_utts)
        all_x_act = act_ftz.featurize_batch(all_system_acts)
        all_x_sys = sys_ftz.featurize_batch(all_system_utts)

        for i in range(len(dg['turns'])):

            # Encode user and action representations
            if args.elmo:
                x_utt = all_x_utt[i].to(device)
                x_sys = all_x_sys[i].to(device)
            else:
                x_utt = [t.to(device) for t in
                         all_x_utt[i]]  # one vector per n
                x_sys = [t.to(device) for t in
                         all_x_sys[i]]  # one vector per n
            x_act = all_x_act[i].to(device)

            featurized_turns.append(Turn(
                all_user_utts[i], all_system_acts[i], all_system_utts[i],
                x_utt, x_act, x_sys,
                all_ys[i], all_lbls[i], all_bsts[i]))

        featurized_dialogs.append(Dialog(featurized_turns))

    return featurized_dialogs


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


class ElmoFeaturizer(Featurizer):

    def __init__(self, elmo, mode):
        super().__init__()
        self.elmo = elmo
        self.mode = mode
        self.map = self.system_act_mapping()

    def featurize_turn(self, turn):
        if self.mode == "utterance":
            turn = ["<bos>"] + turn + ["<eos>"]
        elif self.mode == "act":
            turn = [item for sublist in turn for item in sublist]
            turn = self.clean_act(turn)
        elif self.mode in ["slot", "value"]:
            turn = [turn]
            #turn = self.clean_act(turn)
        # get elmo embeddings
        if not turn:
            turn = [["<NIL>"]]
        e_toks = self.elmo.batch_to_embeddings(turn)[0][0]

        # Sequence of elmo embeddings with all 3 layers concatenated for each token
        #tok_embs = torch.cat((e_toks[0, :, :],
        #                      e_toks[1, :, :],
        #                      e_toks[2, :, :]),
        #                      dim=1)

        # Average 3 ELMo layers
        tok_embs = torch.mean(e_toks, dim=0)

        # max over tokens & flatten
        pooled = torch.max(tok_embs, dim=0)[0].view(-1)
        #pooled = torch.max(e_toks, dim=1)[0].view(-1)

        return tok_embs, pooled

    def featurize_batch(self, batch):
        if self.mode == "utterance":
            batch = [["<bos>"] + turn + ["<eos>"] for turn in batch]
        elif self.mode == "act":
            batch = [self.clean_act([item for sublist in turn for item in sublist]) for turn in batch]
        #elif self.mode in ["slot", "value"]:
        #    batch = [self.clean_act(turn) for turn in batch]

        e_toks = self.elmo.batch_to_embeddings(batch)[0]

        # Sequence of elmo embeddings with all 3 layers concatenated for each token
        #tok_embs = torch.cat((e_toks[:, 0, :, :],
        #                      e_toks[:, 1, :, :],
        #                      e_toks[:, 2, :, :]),
        #                      dim=2)

        # Average 3 ELMo layers
        tok_embs = torch.mean(e_toks, dim=1)

        # max over tokens & flatten
        pooled = torch.max(tok_embs, dim=1)[0].view(len(batch), -1)
        #pooled = torch.max(e_toks, dim=2)[0].view(len(batch), -1)

        return tok_embs, pooled

    def clean_act(self, turn):
        #turn = [self.map.get(item, item) for item in turn]
        turn = ["<bos>"] + turn + ["<eos>"]
        return turn

    def system_act_mapping(self):
        mapping = {'Dest': 'destination',
                   'Ref': 'reference',
                   '=': 'is',
                   'Addr': 'address',
                   '?': 'unknown',
                   'a': 'a'}
        return mapping


class UserInputNgramFeaturizer(Featurizer):

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
        utt_reps = []
        for k in range(self.n):
            kgram = make_n_gram_bow(seq, k + 1, mode='sum')
            utt_reps.append(torch.Tensor(kgram))
        #print(len(turn), turn)
        #print(len(utt_reps), [k.shape for k in utt_reps])
        #return utt_reps[self.n - 1]
        return utt_reps

    def featurize_dialog(self, dialog):
        return [self.featurize_turn(t) for t in dialog.to_dict()['turns']]

    def featurize_batch(self, batch):
        return [self.featurize_turn(t) for t in batch]


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

    def featurize_batch(self, batch):
        return [self.featurize_turn(t) for t in batch]


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

    def featurize_batch(self, batch):
        return [self.featurize_turn(t) for t in batch]


class SlotFeaturizer(Featurizer):

    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings
        oov_len = len(embeddings[list(embeddings.keys())[0]])
        self.oov = np.zeros(oov_len)

    def featurize_turn(self, slot):
        vecs = np.array([self.embeddings.get(w.lower(), self.oov)
                         for w in slot])
        if not len(vecs):
            vecs = [self.oov]
        slot_emb = np.max(vecs, 0)  # max across dimensions
        return torch.Tensor(slot_emb)

    def featurize_batch(self, batch):
        return [self.featurize_turn(t) for t in batch]


class ValueFeaturizer(Featurizer):

    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings
        oov_len = len(embeddings[list(embeddings.keys())[0]])
        self.oov = np.zeros(oov_len)

    def featurize_turn(self, val):
        vecs = np.array([self.embeddings.get(w.lower(), self.oov)
                         for w in val])
        if not len(vecs):
            vecs = [self.oov]
        val_emb = np.max(vecs, 0)  # max across dimensions
        return torch.Tensor(val_emb)

    def featurize_batch(self, batch):
        return [self.featurize_turn(t) for t in batch]
