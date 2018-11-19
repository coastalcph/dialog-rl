import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm
import numpy as np
from tqdm import tqdm
from util import util


# TODO refactor such that encoder classes are declared within StateNet, allows
# for better modularization and sharing of instances/variables such as
# embeddings

class MultiScaleReceptors(nn.Module):
    """

    """
    def __init__(self, in_dim, out_dim, receptors):
        super().__init__()
        # Number of linear networks
        self.receptors = receptors
        # Init linear networks
        for i in range(self.receptors):
            setattr(self, 'linear_out_{}'.format(i), nn.Linear(in_dim, out_dim))

    def forward(self, utt_ngram_rep):
        out = []
        # Get output for every linear network
        for i in range(self.receptors):
            lin_layer = getattr(self, 'linear_out_{}'.format(i))
            out.append(lin_layer(utt_ngram_rep))
        # Return concatenated ngram representation
        return torch.cat(out)


class MultiScaleReceptorsModule(nn.Module):
    """

    """
    def __init__(self, in_dim, out_dim, receptors, n):
        super().__init__()
        self.receptors = receptors
        self.n = n
        #self.layer_norm = LayerNorm(in_dim)
        self.layer_norm = LayerNorm(receptors * out_dim)
        #self.linear_out = nn.Linear(in_dim, out_dim)
        self.linear_out = nn.Linear(receptors * out_dim, out_dim)

        # Initialize the c linear networks for each k-gram utt rep for 1 >= k >= n
        for i in range(n):
            msr_in_dim = in_dim * (i + 1)
            setattr(self, 'linear_out_r{}'.format(i),
                    MultiScaleReceptors(msr_in_dim, out_dim, self.receptors))

    def forward(self, user_ngram_utterances):
        """
        :param user_ngram_utterances:
        :return:
        """
        rets = []
        # For each k-gram utterance representation, get output from MSR networks
        for i in range(self.n):
            msr = getattr(self, 'linear_out_r{}'.format(i))
            msr_out = msr(user_ngram_utterances[i])
            rets.append(msr_out)

        rets = torch.stack(rets)
        out = torch.sum(rets, 0)
        out = self.layer_norm(out)
        out = F.relu(out)
        out = self.linear_out(out)

        return out


class UtteranceEncoder(nn.Module):
    """

    """

    def __init__(self, in_dim, out_dim, receptors):
        super().__init__()
        self.receptors = receptors
        # TODO multiple receptors
        # self.layers = [nn.Linear(in_dim, out_dim) for _ in range(receptors)]
        self.layer_norm = LayerNorm(in_dim)
        self.linear_out = nn.Linear(in_dim, out_dim)

    def forward(self, user_utterance):
        """

        :param user_utterance:
        :return:
        """
        try:
            out = self.layer_norm(user_utterance)
        except RuntimeError:
            print(user_utterance, user_utterance.shape)

        out = F.relu(out)
        out = self.linear_out(out)
        return out


class ActionEncoder(nn.Module):
    """

    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, action):
        """

        :param action:
        :return:
        """
        return F.relu(self.linear(action))


class SlotEncoder(nn.Module):
    """

    """

    def __init__(self, in_dim, out_dim, embeddings):
        """

        :param in_dim:
        :param out_dim:
        """
        super().__init__()
        self.embeddings = embeddings
        self.embeddings_len = len(embeddings.get("i"))
        self.linear = nn.Linear(self.embeddings_len, out_dim)

    def forward(self, slot):
        """

        :param slot:
        :return:
        """
        # remove domain prefix ('restaurant-priceRange' -> 'priceRange')
        domain, slot = slot.split("-", 1)
        # split at uppercase to get vectors ('priceRange' -> ['price', 'range'])
        words = util.split_on_uppercase(slot, keep_contiguous=True)
        vecs = [self.embeddings.get(w.lower()) for w in words]
        if not vecs:
            vecs = [[0 for _ in range(len(self.embeddings.get("i")))]]

        sv = torch.Tensor(vecs)
        sv, _ = torch.max(sv, 0)
        # print(domain, slot, words, len(vecs), sv.shape)

        return F.relu(self.linear(sv))


class PredictionEncoder(nn.Module):
    """

    """

    def __init__(self, in_dim, hidden_dim, out_dim):
        """

        """
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, inputs, hidden):
        """
        Runs the RNN to compute outputs based on history from previous
        slots and turns. We maintain the hidden state across calls to this
        function.
        :param inputs: shape (batch_size, embeddings)
        :param hidden:
        :return: shape (batch_size, self.out_dim)
        """
        batch_size, embedding_length = inputs.view(1, -1).shape
        # reshape input to length 1 sequence (RNN expects input shape
        # [sequence_length, batch_size, embedding_length])
        inputs = inputs.view(1, batch_size, embedding_length)
        # compute output and new hidden state
        rnn_out, hidden = self.rnn(inputs, hidden)
        # reshape to [batch_size,
        rnn_out = rnn_out.view(batch_size, -1)
        o = F.relu(self.linear(rnn_out))
        # print("prediction vector:", o.shape)
        return o, hidden


class ValueEncoder(nn.Module):
    """

    """
    def __init__(self, out_dim, embeddings):
        """

        :param in_dim:
        :param out_dim:
        """
        super().__init__()
        self.embeddings = embeddings
        self.embeddings_len = len(embeddings.get("i"))
        self.linear = nn.Linear(self.embeddings_len, out_dim)

    def forward(self, slot):
        """

        :param slot:
        :return:
        """
        v = self.embeddings.get(slot)
        if not v:
            v = torch.zeros(len(self.embeddings.get("i")))
        v = torch.Tensor(v)
        return F.relu(self.linear(v))


class StateNet(nn.Module):
    """
    Implementation based on Ren et al. (2018): Towards Universal Dialogue
    State Tracking. EMNLP 2018. http://aclweb.org/anthology/D18-1299.

    The paper remains unclear regarding a number of points, for which we
    make decisions based on our intuition. These are, for example:

    (1) How are predictions across turns aggregated on the dialogue level?
        Is the probability for a slot-value pair maxed across turns?
        - We assume yes.
    (2) The paper says that parameters are optimized based on cross-entropy
        between slot-value predictions and gold labels. How does this integrate
        the LSTM that is located outside the turn loop?
        - Not really sure how to handle this yet...
    (3) Is the LSTM updated after every turn AND every slot representation
        computation?
        - We assume yes.

    """

    def __init__(self, input_user_dim, input_action_dim, hidden_dim, receptors,
                 embeddings):
        """

        :param input_user_dim: dimensionality of user input embeddings
        :param input_action_dim: dimensionality of action embeddings
        :param hidden_dim:
        :param receptors:
        :param embeddings:
        """
        super().__init__()
        u_in_dim = input_user_dim
        a_in_dim = input_action_dim
        s_in_dim = input_user_dim
        self.hidden_dim = hidden_dim
        n = int(u_in_dim / a_in_dim)
        self.utt_enc = MultiScaleReceptorsModule(a_in_dim, hidden_dim, receptors, n)
        self.utterance_encoder = UtteranceEncoder(u_in_dim, hidden_dim,
                                                  receptors)
        self.action_encoder = ActionEncoder(a_in_dim, hidden_dim)
        self.slot_encoder = SlotEncoder(s_in_dim, 3*hidden_dim, embeddings)
        self.prediction_encoder = PredictionEncoder(3*hidden_dim, hidden_dim, hidden_dim)
        self.value_encoder = ValueEncoder(hidden_dim, embeddings)
        self.embeddings = embeddings
        self.embeddings_len = len(embeddings.get("i"))
        self.device = self.get_device()

    # @property
    def get_device(self):
        # if self.args.gpu is not None and torch.cuda.is_available():
        #     return torch.device('cuda')
        # else:
        return torch.device('cpu')

    def embed(self, w, numpy=False):
        e = self.embeddings.get(w)
        if not e:
            e = np.zeros(self.embeddings_len)
        if numpy:
            return np.array(e)
        else:
            return torch.Tensor(e)

    def embed_batch(self, b):
        e = [self.embed(w, numpy=True) for w in b]
        return torch.Tensor(e)

    def forward_turn(self, x_user, x_action, x_sys, slots2values, hidden,
                     labels=None):
        """

        :param x_user: shape (batch_size, user_embeddings_dim)
        :param x_action: shape (batch_size, action_embeddings_dim)
        :param x_sys: shape (batch_size, sys_embeddings_dim)
        :param hidden: shape (batch_size, 1, hidden_dim)
        :param slots2values: dict mapping slots to values to be tested
        :param labels: dict mapping slots to one-hot ground truth value
        representations
        :return: tuple (loss, probs, hidden), with `loss' being the overall
        loss across slots, `probs' a dict mapping slots to probability
        distributions over values, `hidden' the new hidden state
        """
        probs = {}

        # Encode user and action representations offline
        #fu = self.utterance_encoder(x_user)  # user input encoding
        fu = self.utt_enc(x_user)  # user ngram input encoding
        fa = self.action_encoder(x_action)  # action input encoding
        #fy = self.utterance_encoder(x_sys)
        fy = self.utt_enc(x_sys)

        # iterate over slots and values, compute probabilities
        for slot, values in slots2values.items():
            # compute encoding of inputs as described in StateNet paper, Sec. 2
            fs = self.slot_encoder(slot).view(-1)  # slot encoding
            # i = torch.cat((fu, fa), 0)
            # i = F.mul(fs, i)
            i = F.mul(fs, torch.cat((fu, fa, fy), 0))  # inputs encoding
            o, hidden = self.prediction_encoder(i, hidden)

            # get probability distribution over values...
            probs[slot] = torch.zeros(len(values))
            for v, value in enumerate(values):
                venc = self.value_encoder(value)
                # ... by computing 2-Norm distance according to paper, Sec. 2.6
                probs[slot][v] = -torch.dist(o, venc)
            probs[slot] = F.softmax(probs[slot], 0)  # softmax it!

        if self.training:
            loss = 0
            for slot in labels.keys():
                # print(slot, probs.keys(), labels.keys())
                loss += F.binary_cross_entropy(probs[slot], labels[slot])
        else:
            loss = torch.Tensor([0]).to(self.device)

        return loss, probs, hidden

    def forward(self, turns, slots2values):
        """

        :param turns:
        :param slots2values:
        :return:
        """
        hidden = torch.zeros(1, 1, self.hidden_dim)
        global_probs = {}
        global_loss = torch.Tensor([0]).to(self.device)

        for x_user, x_action, x_sys, labels in turns:
            loss, turn_probs, hidden = self.forward_turn(x_user, x_action,
                                                         x_sys, slots2values,
                                                         hidden, labels)

            global_loss += loss
            for slot, values in slots2values.items():
                global_probs[slot] = torch.zeros(len(values))
                for v, value in enumerate(values):
                    global_probs[slot][v] = max(global_probs[slot][v],
                                                turn_probs[slot][v])

        # get final predictions
        ys = {}
        for slot, probs in global_probs.items():
            score, argmax = probs.max(0)
            ys[slot] = int(argmax)

        return ys, global_loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(path)

