import torch
from torch import nn
from torch.nn import functional as F


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
        :param inputs:
        :param hidden:
        :return:
        """
        batch_size, embedding_length = inputs.shape
        # reshape input to length 1 sequence (RNN expects input shape
        # [sequence_length, batch_size, embedding_length])
        inputs = inputs.view(1, batch_size, embedding_length)
        # compute output and new hidden state
        rnn_out, hidden = self.rnn(inputs, hidden)
        # reshape to [batch_size,
        rnn_out = rnn_out.view(batch_size, -1)
        o = F.relu(self.linear(rnn_out))
        print(o.shape)
        return o, hidden


class ValueEncoder(nn.Module):
    """

    """

    def __init__(self, in_dim):
        """

        """
        super().__init__()
        self.in_dim = in_dim

    def forward(self, value):
        """

        :param value:
        :return:
        """
        v = torch.randn(1, self.in_dim)  # TODO
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
        self.hidden_dim = hidden_dim
        self.user_utterance_encoder = UserUtteranceEncoder(u_in_dim, hidden_dim,
                                                           receptors)
        self.action_encoder = ActionEncoder(a_in_dim, hidden_dim)
        self.slot_encoder = SlotEncoder(s_in_dim, 2*hidden_dim)
        self.prediction_encoder = PredictionEncoder(2*hidden_dim, 2*hidden_dim, s_in_dim)
        self.value_encoder = ValueEncoder(s_in_dim)

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
