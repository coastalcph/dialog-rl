import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from stanza.nlp.corenlp import CoreNLPClient
from nltk import word_tokenize
from pprint import pprint

client = None

sys_act_map = {'Dest': 'destination',
               'Ref': 'reference',
               '=': 'is',
               'Addr': 'address',
               '?': 'unknown',
               'a': 'a'}



def annotate(sent):
    global client
    if client is None:
        client = CoreNLPClient(default_annotators='ssplit,tokenize'.split(','))
    words = []
    for sent in client.annotate(sent).sentences:
        for tok in sent:
            words.append(tok.word)
    return words


class Turn:

    def __init__(self, turn_id, transcript, turn_label, belief_state, system_acts, system_transcript, num=None):
        self.id = turn_id
        self.transcript = transcript
        self.turn_label = turn_label
        self.belief_state = belief_state
        self.system_acts = system_acts
        self.system_transcript = word_tokenize(system_transcript)
        self.num = num or {}

    def to_dict(self, elmo=False):
        if elmo:
            return {'turn_id': self.id,
                    'transcript': self.transcript,
                    'turn_label': self.turn_label,
                    'belief_state': self.belief_state,
                    'system_acts': self.system_acts,
                    'system_transcript': self.system_transcript,
                    #'usr_trans_elmo': self.usr_trans_elmo,
                    'usr_trans_elmo_pool': self.usr_trans_elmo_pool,
                    #'sys_trans_elmo': self.sys_trans_elmo,
                    'sys_trans_elmo_pool': self.sys_trans_elmo_pool,
                    #'sys_acts_elmo': self.sys_acts_elmo,
                    'sys_acts_elmo_pool': self.sys_acts_elmo_pool}
        else:
            return {'turn_id': self.id,
                    'transcript': self.transcript,
                    'turn_label': self.turn_label,
                    'belief_state': self.belief_state,
                    'system_acts': self.system_acts,
                    'system_transcript': self.system_transcript}
                    #, 'num': self.num}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def annotate_raw(cls, raw):
        system_acts = []
        for a in raw['system_acts']:
            if isinstance(a, list):
                s, v = a
                system_acts.append(['inform'] + s.split() + ['='] + v.split())
            else:
                system_acts.append(['request'] + a.split())
        # NOTE: fix inconsistencies in data label
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
        return cls(
            turn_id=raw['turn_idx'],
            transcript=annotate(raw['transcript']),
            system_acts=system_acts,
            turn_label=[[fix.get(s.strip(), s.strip()), fix.get(v.strip(), v.strip())] for s, v in raw['turn_label']],
            belief_state=raw['belief_state'],
            system_transcript=raw['system_transcript'],
        )

    def numericalize_(self, vocab):
        self.num['transcript'] = vocab.word2index(['<sos>'] + [w.lower() for w in self.transcript + ['<eos>']], train=True)
        self.num['system_acts'] = [vocab.word2index(['<sos>'] + [w.lower() for w in a] + ['<eos>'], train=True) for a in self.system_acts + [['<sentinel>']]]


class Dialogue:

    def __init__(self, dialogue_id, turns, domain, elmo=None):
        self.id = dialogue_id
        self.turns = turns
        self.domain = domain
        self.elmo = elmo

    def __len__(self):
        return len(self.turns)

    def to_dict(self):
        return { 'dialogue_id': self.id, 'turns': [t.to_dict(elmo=self.elmo) for t in self.turns],'domain': self.domain}

    def to_elmo(self, elmo):

        self.elmo = True

        utt, sys = elmo
        usr_utts, sys_utts, sys_acts = [], [], []
        for turn in self.turns:
            # Fix system acts to more natural language
            turn.system_acts = [[sys_act_map.get(t, t) for t in tu] for tu in turn.system_acts]

            # Batch utterances and system acts
            usr_utts.append(turn.transcript)
            sys_utts.append(turn.system_transcript)
            sys_acts.append(turn.system_acts)

        # Featurize
        usr_embs, pooled_usr = utt.featurize_batch(usr_utts)
        sys_embs, pooled_sys = utt.featurize_batch(sys_utts)
        sys_act_embs, sys_act_pooled = sys.featurize_batch(sys_acts)

        # Add both list of ELMO embs for each token and a pooled one with different key
        for i, turn in enumerate(self.turns):
            #setattr(turn, 'usr_trans_elmo', usr_embs[i])
            setattr(turn, 'usr_trans_elmo_pool', pooled_usr[i])
            #setattr(turn, 'sys_trans_elmo', sys_embs[i])
            setattr(turn, 'sys_trans_elmo_pool', pooled_sys[i])
            #setattr(turn, 'sys_acts_elmo', sys_act_embs[i])
            setattr(turn, 'sys_acts_elmo_pool', sys_act_pooled[i])

        return self

    @classmethod
    def from_dict(cls, d):
        return cls(d['dialogue_id'], [Turn.from_dict(t) for t in d['turns']], d['domain'])

    @classmethod
    def annotate_raw(cls, raw):
        return cls(raw['dialogue_idx'], [Turn.annotate_raw(t) for t in raw['dialogue']], raw['domain'])


class Dataset:

    def __init__(self, dialogues):
        self.dialogues = dialogues

    def __len__(self):
        return len(self.dialogues)

    def iter_dialogs(self):
        for d in self.dialogues:
            yield d

    def iter_turns(self):
        for d in self.dialogues:
            for t in d.turns:
                yield t

    def to_elmo(self, elmo):
        dialogues_elmo = []
        for d in tqdm(self.dialogues, desc='Adding ELMo features'):
            e = d.to_elmo(elmo)
            dialogues_elmo.append(e)
        self.dialogues = dialogues_elmo

    def to_dict(self):
        return {'dialogues': [d.to_dict() for d in self.dialogues]}

    @classmethod
    def from_dict(cls, d, elmo=None):
        return cls([Dialogue.from_dict(dd) for dd in tqdm(d['dialogues'])])


    def annotate_raw(cls, fname):
        with open(fname) as f:
            data = json.load(f)
            return cls([Dialogue.annotate_raw(d) for d in tqdm(data)])

    def numericalize_(self, vocab):
        for t in self.iter_turns():
            t.numericalize_(vocab)
    def extract_ontology(self):
        slots = set()
        values = defaultdict(set)
        for t in self.iter_turns():
            for s, v in t.turn_label:
                slots.add(s)
                values[s].add(v.lower())

        return Ontology(sorted(list(slots)), {k: sorted(list(v)) for k, v in values.items()})


    def batch(self, batch_size, shuffle=False, whole_dialogs=False):
        if whole_dialogs:
            iter_items = list(self.iter_dialogs())
        else:
            iter_items = list(self.iter_turns())
        if shuffle:
            np.random.shuffle(iter_items )
        for i in tqdm(range(0, len(iter_items ), batch_size)):
            yield iter_items [i:i+batch_size]

    def evaluate_preds(self, preds):
        request = []
        inform = []
        joint_goal = []
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
        i = 0
        for d in self.dialogues:
            pred_state = {}
            for t in d.turns:
                gold_request = set([(s, v) for s, v in t.turn_label if s == 'request'])
                gold_inform = set([(s, v) for s, v in t.turn_label if s != 'request'])
                pred_request = set([(s, v) for s, v in preds[i] if s == 'request'])
                pred_inform = set([(s, v) for s, v in preds[i] if s != 'request'])
                request.append(gold_request == pred_request)
                inform.append(gold_inform == pred_inform)

                gold_recovered = set()
                pred_recovered = set()
                for s, v in pred_inform:
                    pred_state[s] = v
                for b in t.belief_state:
                    for s, v in b['slots']:
                        if b['act'] != 'request':
                            gold_recovered.add((b['act'], fix.get(s.strip(), s.strip()), fix.get(v.strip(), v.strip())))
                for s, v in pred_state.items():
                    pred_recovered.add(('inform', s, v))
                joint_goal.append(gold_recovered == pred_recovered)
                i += 1
        return {'turn_inform': np.mean(inform), 'turn_request': np.mean(request), 'joint_goal': np.mean(joint_goal)}

    def record_preds(self, preds, to_file):
        data = self.to_dict()
        i = 0
        for d in data['dialogues']:
            for t in d['turns']:
                t['pred'] = sorted(list(preds[i]))
                i += 1
        with open(to_file, 'wt') as f:
            json.dump(data, f)


class Ontology:

    def __init__(self, slots=None, values=None, num=None):
        self.slots = slots or []
        self.values = values or {}
        self.num = num or {}

    def __add__(self, another):
        new_slots = sorted(list(set(self.slots + another.slots)))
        new_values = {s: sorted(list(set(self.values.get(s, []) + another.values.get(s, [])))) for s in new_slots}
        return Ontology(new_slots, new_values)

    def __radd__(self, another):
        return self if another == 0 else self.__add__(another)

    def to_dict(self):
        return {'slots': self.slots, 'values': self.values, 'num': self.num}

    def numericalize_(self, vocab):
        self.num = {}
        for s, vs in self.values.items():
            self.num[s] = [vocab.word2index(annotate('{} = {}'.format(s, v)) + ['<eos>'], train=True) for v in vs]

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
