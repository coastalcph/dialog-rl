import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict


class UserUtteranceEncoder(nn.Module):
    """

    """

    def __init__(self, in_dim, out_dim, receptors):
        super().__init__()
        self.receptors = receptors
        self.layers = [nn.Linear(in_dim, out_dim) for _ in range(receptors)]
        self.linear_out = nn.Linear(in_dim, out_dim)

    def forward(self, user_utterance):
        """

        :param user_utterance:
        :return:
        """
        added_ngrams = torch.sum(torch.stack(user_utterance), 0)
        out = added_ngrams

        # out = nn.LayerNorm(out)
        out = F.relu(out)
        out = self.linear_out(out)
        print(out, out.shape)
        return out
        # return self.linear_out(F.relu(nn.LayerNorm(added_ngrams)))


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

    def __init__(self, in_dim, out_dim):
        """

        :param in_dim:
        :param out_dim:
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, slot):
        """

        :param slot:
        :return:
        """
        sv = torch.randn(1, self.linear.in_features)  # TODO retrieve slot vector
        return F.relu(self.linear(sv))


class PredictionEncoder(nn.Module):
    """

    """

    def __init__(self):
        """

        """
        super().__init__()

    def forward(self, inputs, hidden):
        """

        :param inputs:
        :param hidden:
        :return:
        """
        # TODO
        o = 0
        return o, hidden


class ValueEncoder(nn.Module):
    """

    """

    def __init__(self):
        """

        """
        super().__init__()

    def forward(self, value):
        """

        :param value:
        :return:
        """

        v = 0
        return v


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

    def __init__(self, featurizers, hidden_dim, receptors):
        """

        :param featurizers:
        :param hidden_dim:
        :param receptors:
        """
        super().__init__()
        u_in_dim = featurizers['user'].dimensionality
        a_in_dim = featurizers['actions'].dimensionality
        s_in_dim = featurizers['slots'].dimensionality
        self.user_utterance_encoder = UserUtteranceEncoder(u_in_dim, hidden_dim,
                                                           receptors)
        self.hidden_dim = hidden_dim
        self.action_encoder = ActionEncoder(a_in_dim, hidden_dim)
        self.slot_encoder = SlotEncoder(s_in_dim, 2*hidden_dim)
        self.prediction_encoder = PredictionEncoder()
        self.value_encoder = ValueEncoder()

    def forward_turn(self, x_user, x_action, slots2values, hidden, labels=None):
        """

        :param x_user:
        :param x_action:
        :param hidden:
        :param slots2values:
        :param labels:
        :return:
        """
        probs = {}

        for slot, values in slots2values.items():
            # compute encoding of inputs as described in StateNet paper, Sec. 2
            fu = self.user_utterance_encoder(x_user)  # user input encoding
            fa = self.action_encoder(x_action)  # action input encoding
            fs = self.slot_encoder(slot)  # slot encoding
            i = F.mul(fs, torch.cat((fu, fa), 1))  # inputs encoding
            o, hidden = self.prediction_encoder(i, hidden)

            # get probability distribution over values...
            probs[slot] = torch.zeros(len(values))
            for v, value in enumerate(values):
                venc = self.value_encoder(value)
                # ... by computing 2-Norm distance according to paper, Sec. 2.6
                probs[slot][v] = -torch.dist(o, venc)
            probs[slot] = F.softmax(probs[slot])  # softmax it!

        if self.training:
            loss = 0
            for slot in slots2values.keys():
                loss += F.cross_entropy(probs[slot], labels[slot])
        else:
            loss = torch.Tensor([0]).to(self.device)

        return loss, probs, hidden

    def forward(self, turns, slots2values):
        """

        :param turns:
        :param slots2values:
        :return:
        """
        hidden = torch.randn(self.hidden_dim)
        global_probs = {}

        for x_user, x_action, labels in turns:
            _, turn_probs, hidden = self.forward_turn(x_user, x_action,
                                                      slots2values, hidden,
                                                      labels)
            for slot, values in slots2values.items():
                global_probs[slot] = torch.zeros(len(values))
                for v, value in enumerate(values):
                    global_probs[slot][v] = max(global_probs[slot][v],
                                                turn_probs[slot][v])

        # get final predictions
        ys = {}
        for slot, probs in global_probs.items():
            score, argmax = probs.max(0)
            ys[slot] = argmax

        return ys
