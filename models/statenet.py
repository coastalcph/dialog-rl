import os
import re
import logging
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from pprint import pformat
from util.eval import evaluate_preds, get_reward
from util import util


# TODO refactor such that encoder classes are declared within StateNet, allows
# for better modularization and sharing of instances/variables such as
# embeddings


eps = np.finfo(np.float32).eps.item()


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
        # Return concatenated ngram representation: stack along receptors,
        # transpose such that output shape is [batch_size, receptors, out_dim]
        return torch.stack(out).transpose(0, 1)


class MultiScaleReceptorsModule(nn.Module):
    """

    """
    def __init__(self, in_dim, out_dim, receptors, n):
        super().__init__()
        self.receptors = receptors
        self.n = n
        self.layer_norm = LayerNorm(receptors * out_dim)
        self.linear_out = nn.Linear(receptors * out_dim, out_dim)

        # Initialize the c linear nets for each k-gram utt rep for 1 >= k >= n
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
        batch_size = len(user_ngram_utterances[0])
        # For each k-gram utterance representation, get output from MSR networks
        for i in range(self.n):
            msr = getattr(self, 'linear_out_r{}'.format(i))
            msr_out = msr(user_ngram_utterances[i])
            rets.append(msr_out)

        rets = torch.stack(rets).transpose(0, 1)
        # sum along n-grams and flatten receptors and hidden
        out = torch.sum(rets, 1).view(batch_size, -1)
        out = self.layer_norm(out)
        out = F.relu(out)
        out = self.linear_out(out)
        return out


class UtteranceEncoder(nn.Module):
    """

    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer_norm = LayerNorm(in_dim)
        self.linear_out = nn.Linear(in_dim, out_dim)

    def forward(self, user_utterance):
        """

        :param user_utterance:
        :return:
        """
        user_utterance = torch.stack(user_utterance).transpose(0, 1)
        try:
            out = self.layer_norm(user_utterance)
        except RuntimeError:
            print(user_utterance, user_utterance.shape)
            raise RuntimeError

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

    def __init__(self, in_dim, out_dim, device):
        """

        :param in_dim:
        :param out_dim:
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.device = device

    def forward(self, slot_embedding):
        """

        :param slot_embedding: Vector representation of slot
        :return:
        """
        return F.relu(self.linear(slot_embedding)).view(-1).to(self.device)


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
        batch_size, embedding_length = inputs.shape
        # reshape input to length 1 sequence (RNN expects input shape
        # [sequence_length, batch_size, embedding_length])
        inputs = inputs.view(1, batch_size, embedding_length)
        # compute output and new hidden state
        # print(hidden.shape, inputs.shape)
        rnn_out, hidden = self.rnn(inputs, hidden)
        # reshape to [batch_size,
        rnn_out = rnn_out.view(batch_size, -1)
        o = F.relu(self.linear(rnn_out))
        # print("prediction vector:", o.shape)
        return o, hidden


class ValueEncoder(nn.Module):
    """

    """
    def __init__(self, in_dim, out_dim, device):
        """

        :param in_dim:
        :param out_dim:
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.device = device

    def forward(self, value_embedding):
        """

        :param value_embedding:
        :return:
        """
        return F.relu(self.linear(value_embedding)).to(self.device)


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

    def __init__(self, input_user_dim, input_action_dim, input_slot_dim,
                 input_value_dim, hidden_dim, receptors,
                 args):
        """

        :param input_user_dim: dimensionality of user input embeddings
        :param input_action_dim: dimensionality of action embeddings
        :param hidden_dim:
        :param receptors:
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.args = args
        self.device = self.get_device(args.gpu)

        if args.encode_sys_utt:
            slot_hidden_dim = 3 * hidden_dim
        else:
            slot_hidden_dim = 2 * hidden_dim

        if not args.elmo:
            input_user_dim = input_user_dim * args.M
        u_in_dim = input_user_dim
        a_in_dim = input_action_dim
        # s_in_dim = input_user_dim
        n = int(u_in_dim / a_in_dim)
        if args.elmo:
            self.utt_enc = UtteranceEncoder(u_in_dim, hidden_dim)
        else:
            self.utt_enc = MultiScaleReceptorsModule(a_in_dim, hidden_dim,
                                                     receptors, n)
        self.action_encoder = ActionEncoder(a_in_dim, hidden_dim)
        self.slot_encoder = SlotEncoder(input_slot_dim, hidden_dim,
                                        self.device)
        self.value_encoder = ValueEncoder(input_value_dim, hidden_dim,
                                          self.device)
        # self.prediction_encoder = PredictionEncoder(slot_hidden_dim,
        #                                             hidden_dim, hidden_dim)
        self.turn_history_rnn = PredictionEncoder(slot_hidden_dim, hidden_dim,
                                                  hidden_dim)
        self.slot_fill_indicator = nn.Linear(hidden_dim, 1)
        self.optimizer = None
        self.epochs_trained = 0
        self.logger = self.get_train_logger()
        self.logger.setLevel(args.log_level.upper())
        self.logger.info(args)
        self.set_optimizer()

    def set_epochs_trained(self, e):
        self.epochs_trained = e

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        # self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr)

    def get_train_logger(self):
        logger = logging.getLogger(
            'train-{}'.format(self.__class__.__name__))
        formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] '
                                      '[%(levelname)-5.5s]  %(message)s')
        file_handler = logging.FileHandler(
            os.path.join(self.args.dout, 'train.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def forward_turn(self, batch, slots2values, hidden):
        """

        # :param x_user: shape (batch_size, user_embeddings_dim)
        # :param x_action: shape (batch_size, action_embeddings_dim)
        # :param x_sys: shape (batch_size, sys_embeddings_dim)
        :param batch:
        :param hidden: shape (batch_size, 1, hidden_dim)
        :param slots2values: dict mapping slots to values to be tested
        # :param labels: dict mapping slots to one-hot ground truth value
        representations
        :return: tuple (loss, probs, hidden), with `loss' being the overall
        loss across slots, `probs' a dict mapping slots to probability
        distributions over values, `hidden' the new hidden state
        """
        batch_size = len(batch)
        probs = defaultdict(list)
        binary_filling_probs = {}

        # user input encoding [batch_size, hidden_dim]

        all_utt = [torch.stack([turn.x_utt[k] for turn in batch])
                   for k in range(len(batch[0].x_utt))]
        fu = self.utt_enc(all_utt)
        # system act input encoding [batch_size, hidden_dim]
        all_act = torch.stack([turn.x_act for turn in batch])
        fa = self.action_encoder(all_act)

        if self.args.encode_sys_utt:
            fy = self.utt_enc(torch.Tensor([turn.x_sys for turn in batch]))
            f_turn_inputs = torch.cat((fu, fa, fy), 1)
        else:
            f_turn_inputs = torch.cat((fu, fa), 1)

        # turn encodings [batch_size, hidden_dim]
        # and RNN hidden state [batch_size, hidden_dim_rnn]
        f_turn, hidden = self.turn_history_rnn(f_turn_inputs, hidden)

        # keep track of number of loss updates for later averaging
        loss_updates = torch.Tensor([0]).to(self.device)

        # iterate over slots and values, compute probabilities
        for slot_id in sorted(slots2values.keys()):
            slot = slots2values[slot_id]
            # compute encoding of inputs as described in StateNet paper, Sec. 2
            fs = self.slot_encoder(slot.embedding)
            # encoding of slot with turns in batch: [batch_size, hidden_dim]
            f_slot_turn = F.mul(fs, f_turn)

            # get binary prediction for slot presence {slot_id: [batch_size, 1]}
            binary_filling_probs[slot_id] = torch.sigmoid(
                self.slot_fill_indicator(f_slot_turn))

            # get probability distribution over values...
            values = slot.values
            for t, turn in enumerate(batch):
                probs[slot_id].append(None)
                if binary_filling_probs[slot_id][t] > 0.5:
                    probs[slot_id][t] = torch.zeros(len(values))
                    for v, value in enumerate(values):
                        venc = self.value_encoder(value.embedding)
                        # by computing 2-Norm distance following paper, Sec. 2.6
                        probs[slot_id][t][v] = -torch.dist(f_slot_turn, venc)

                    # softmax it!
                    probs[slot_id][t] = F.softmax(probs[slot_id][t], 0)

        loss = torch.Tensor([0]).to(self.device)
        if self.training:
            for slot_id in slots2values.keys():

                # loss for binary slot presence
                # gold: 1 if slot in turn.labels (meaning it's filled), else 0
                # [batch_size, 1]
                if binary_filling_probs[slot_id] is not None:
                    gold_slot_filling = torch.Tensor(
                        [float(slot_id in turn.labels) for turn in batch]
                    ).view(-1, 1).to(self.device)
                    loss += self.args.eta * F.binary_cross_entropy(
                        binary_filling_probs[slot_id],
                        gold_slot_filling).to(self.device)
                    loss_updates += 1

                for t, turn in enumerate(batch):
                    # loss for slot-value pairing, if slot is present
                    if slot_id in turn.labels and \
                            binary_filling_probs[slot_id][t] > 0.5:
                        loss += F.binary_cross_entropy(
                            probs[slot_id][t],
                            turn.labels[slot_id]
                        ).to(self.device)
                        loss_updates += 1

        loss = loss / loss_updates
        mean_slots_filled = len(probs) / len(slots2values)
        return loss, probs, hidden, mean_slots_filled

    def forward(self, dialogs, slots2values):
        batch_size = len(dialogs)
        hidden = torch.zeros(1, batch_size, self.hidden_dim).to(self.device)
        global_probs = [{} for _ in range(batch_size)]
        global_loss = torch.Tensor([0]).to(self.device)
        per_turn_mean_slots_filled = []
        ys_turn = [[] for _ in range(batch_size)]
        scores = [defaultdict(list) for _ in range(batch_size)]

        batch_turns_first, mask = util.turns_first(dialogs)
        max_turns = max(mask)
        # turns is list of t-th turns in current batch
        for t, turns in enumerate(batch_turns_first):
            loss, turn_probs, hidden, mean_slots_filled = \
                self.forward_turn(turns, slots2values, hidden)
            global_loss += loss
            turn_probs = util.invert_slot_turns(turn_probs, batch_size)
            turn_preds = [{} for _ in range(batch_size)]
            for batch_item in range(batch_size):
                if t < mask[batch_item]:
                    for slot_id, slot in slots2values.items():
                        if turn_probs[batch_item][slot_id] is not None:

                            global_probs[batch_item][slot_id] = \
                                torch.zeros(len(slot.values))
                            argmax = np.argmax(turn_probs[batch_item][slot_id].
                                               detach().numpy(), 0)
                            turn_preds[batch_item][slot_id] = \
                                slots2values[slot_id].values[int(argmax)].value
                            for v, value in enumerate(slot.values):
                                global_probs[batch_item][slot_id][v] = max(
                                    global_probs[batch_item][slot_id][v],
                                    turn_probs[batch_item][slot_id][v])
                            scores[batch_item][slot_id].append(
                                turn_probs[batch_item][slot_id])

                    ys_turn[batch_item].append(turn_preds[batch_item])

        ys = [{} for _ in range(batch_size)]
        for batch_item in range(batch_size):
            for slot, probs in global_probs[batch_item].items():
                score, argmax = probs.max(0)
                ys[batch_item][slot] = slots2values[slot].values[
                    int(argmax)].value

        global_loss = global_loss / max_turns
        if per_turn_mean_slots_filled:
            dialog_mean_slots_filled = np.mean(per_turn_mean_slots_filled)
        else:
            dialog_mean_slots_filled = 0.0
        return ys, ys_turn, scores, global_loss, dialog_mean_slots_filled

    def run_train(self, dialogs_train, dialogs_dev, s2v, args,
                  early_stopping=None):
        track = defaultdict(list)
        if self.optimizer is None:
            self.set_optimizer()
        self.logger.info("Starting training...")
        if torch.cuda.is_available() and self.device.type == 'cuda':
            s2v = util.s2v_to_device(s2v, self.device)
        best = {}
        iteration = 0
        no_improvements_for = 0
        for epoch in range(1, args.epochs+1):
            global_mean_slots_filled = []
            # logger.info('starting epoch {}'.format(epoch))

            if not hasattr(self, "epochs_trained"):
                self.set_epochs_trained(0)
            self.epochs_trained += 1

            # train and update parameters
            self.train()
            train_predictions = []
            s = []
            for batch in tqdm(list(util.make_batches(dialogs_train,
                                                     args.batch_size))):
                iteration += 1
                self.zero_grad()
                predictions, turn_predictions, scores, loss, mean_slots_filled = \
                    self.forward(batch, s2v)
                for i in range(len(batch)):
                    train_predictions.append((predictions[i],
                                              turn_predictions[i]))
                # print(turn_predictions)
                global_mean_slots_filled.append(mean_slots_filled)
                loss.backward()
                self.optimizer.step()
                track['loss'].append(loss.item())

            # evalute on train and dev
            summary = {'iteration': iteration, 'epoch': self.epochs_trained}
            for k, v in track.items():
                summary[k] = sum(v) / len(v)
            self.logger.info("Evaluating...")
            predictions, turn_predictions = zip(*train_predictions)
            summary.update({'eval_train_{}'.format(k): v for k, v in
                            evaluate_preds(dialogs_train, predictions,
                                           turn_predictions, args.eval_domains
                                           ).items()})
            summary.update({'eval_dev_{}'.format(k): v for k, v in
                            self.run_eval(dialogs_dev, s2v, args.eval_domains,
                                          self.args.dout +
                                          "/prediction_dv_{}.json".format(epoch)
                                          ).items()})

            #global_mean_slots_filled = np.mean(global_mean_slots_filled)
            #self.logger.info("Predicted {}% slots as present".format(
            #    global_mean_slots_filled*100))
            self.logger.info("Epoch summary: " + str(summary))

            # do early stopping saves
            stop_key = 'eval_dev_{}'.format(args.stop)
            train_key = 'eval_train_{}'.format(args.stop)
            if best.get(stop_key, 0) <= summary[stop_key]:
                no_improvements_for = 0
                best_dev = '{:f}'.format(summary[stop_key])
                best_train = '{:f}'.format(summary[train_key])
                best.update(summary)
                self.save(best,
                          identifier='epoch={epoch},iter={iteration},'
                                     'train_{key}={train},dev_{key}={dev}'
                                     ''.format(
                                        epoch=self.epochs_trained,
                                        iteration=iteration,
                                        train=best_train, dev=best_dev,
                                        key=args.stop)
                          )
                self.prune_saves()
            else:
                no_improvements_for += 1
                if no_improvements_for > args.patience:
                    self.logger.info("Ending training after model did not"
                                     "improve for {} epochs".format(
                                                no_improvements_for))
                    break
                else:
                    self.logger.info("Model did not improve for {} epochs. "
                                     "Patience is {} epochs.".format(
                                        no_improvements_for, args.patience))

            summary.update({'best_{}'.format(k): v for k, v in best.items()})
            self.logger.info(pformat(summary))
            track.clear()

    def run_train_reinforce(self, dialogs_train, dialogs_dev, s2v, args, baseline=None):
        nn.utils.clip_grad_norm(self.parameters(), 5)
        track = defaultdict(list)
        if self.optimizer is None:
            self.set_optimizer()
        self.logger.info("Starting reinforcement training...")
        if torch.cuda.is_available() and self.device.type == 'cuda':
            s2v = util.s2v_to_device(s2v, self.device)
        # Lower learning rate for reinforcement training
        if args.resume:
            lr_rl = args.lr * 0.1
            print('REINFORCE lr: {}'.format(lr_rl) )
            self.optimizer = optim.Adam(self.parameters(), lr=lr_rl)

        best = {}
        iteration = 0
        no_improvements_for = 0
        epoch_rewards = []
        epoch_jg = []

        best_reward = 0
        hill_climb_patience = 0
        for epoch in range(1, args.epochs + 1):
            global_mean_slots_filled = []
            # logger.info('starting epoch {}'.format(epoch))

            if not hasattr(self, "epochs_trained"):
                self.set_epochs_trained(0)
            self.epochs_trained += 1

            # train and update parameters
            self.train()
            train_predictions = []
            for batch in tqdm(list(util.make_batches(dialogs_train,
                                                     args.batch_size))):
                batch_rewards = []
                batch_base_rewards = []
                batch_scores = []
                entropies = []
                batch_losses = []
                iteration += 1
                self.zero_grad()

                predictions, turn_predictions, scores, loss, mean_slots_filled = \
                    self.forward(batch, s2v)

                eval_scores = evaluate_preds(batch, predictions, turn_predictions,
                                             args.eval_domains)

                reward = get_reward(eval_scores)
                batch_reward = reward
                #scale = (-5, 5)
                #batch_reward = shape_reward(reward, scale_out=scale)

                # Eval on dev set and roll back if performance has gone down too much
                if iteration % 5 == 0:
                    # Dev predictions
                    dev_rew = self.run_eval(dialogs_dev, s2v, args.eval_domains, None)['dialog_reward']#['dialog_reward']
                    self.train()

                    if dev_rew > best_reward:
                        print('Current best rew:', dev_rew)
                        best_reward = dev_rew
                        self.save('best-rl', 'best-rl')
                    elif hill_climb_patience == 15:
                        print('Patience reached, rolling back to previous best')
                        fname = self.args.dout + '/best-rl.t7'
                        self.load(fname)
                        #self.optimizer.state['epoch'] = self.epochs_trained
                        hill_climb_patience = 0

                        # Get new predictions/reward
                        predictions, turn_predictions, scores, loss, mean_slots_filled = \
                            self.forward(batch, s2v)
                        eval_scores = evaluate_preds(batch, predictions, turn_predictions,
                                                     args.eval_domains)
                        reward = get_reward(eval_scores)
                        batch_reward = reward
                    else:
                        hill_climb_patience += 1

                base_reward = None
                if baseline:
                    base_preds, base_turn_preds, _, _, _ = \
                        baseline.forward(batch, s2v)
                    base_eval_scores = evaluate_preds(batch, base_preds, base_turn_preds,
                                                 args.eval_domains)
                    b_reward = get_reward(base_eval_scores)
                    # Fiddle around with baseline reward scaling
                    base_reward = b_reward #shape_reward(b_reward, scale_out=scale)

                """
                for batch_item, slot2score in enumerate(scores):
                    for slot, score in slot2score.items():
                        for t in range(len(score)):
                            slot_turn_scores = F.softmax(scores[batch_item][slot][t])
                            m = Categorical(slot_turn_scores)
                            slot_turn_prediction = m.sample()
                            slot_turn_prediction_log_prob = m.log_prob(
                                slot_turn_prediction)
                            # Entropy
                            entropy = (slot_turn_prediction_log_prob * torch.exp(slot_turn_prediction_log_prob))
                            entropies.append(entropy)
                            # credit assignment: copy dialog-level reward to each
                            # slot/turn
                            batch_rewards.append(batch_reward)
                            batch_scores.append(slot_turn_prediction_log_prob)
                """
                for batch_item, slot2score in enumerate(scores):
                    # Rewards, log probs and entropy for s-v pairs in each turn
                    rews = []
                    brews = []
                    prbs = []
                    ents = []
                    for slot, score in slot2score.items():
                        for t in range(len(score)):
                            # Sample and compute log prob for slot predictions for turn
                            slot_turn_scores = scores[batch_item][slot][t] #F.softmax(scores[batch_item][slot][t])
                            m = Categorical(slot_turn_scores)
                            slot_turn_prediction = m.sample()
                            slot_turn_prediction_log_prob = m.log_prob(
                                slot_turn_prediction)
                            # TODO take argmax with prob 1-eps and sample with prob eps

                            # Entropy
                            entropy = (slot_turn_prediction_log_prob * torch.exp(slot_turn_prediction_log_prob))
                            # credit assignment: copy dialog-level reward to each
                            # slot/turn for discounted reward
                            ents.append(entropy)
                            rews.append(batch_reward)
                            brews.append(base_reward)
                            prbs.append(slot_turn_prediction_log_prob)

                            #entropies.append(entropy)
                            #batch_rewards.append(batch_reward)
                            #batch_base_rewards.append(base_reward)
                            #batch_scores.append(slot_turn_prediction_log_prob)

                    # Compute losses for batch item
                    bi_loss = self.reinforce_loss(rews, prbs, brews, ents, self.args.gamma)
                    batch_losses.append(bi_loss)

                #print(len(scores))
                global_mean_slots_filled.append(mean_slots_filled)
                track['loss'].append(loss.item())

                #if batch_rewards:
                #    self.reinforce_update(batch_rewards, batch_scores,
                #                          self.args.gamma, batch_base_rewards, entropies)

                if batch_losses:
                    self.reinforce_update_losses(batch_losses)

                #if iteration % 10 == 0:
                #    ev = self.run_eval(dialogs_dev, s2v, args.eval_domains, None)
                #    self.train()
                #    epoch_rewards.append(ev['dialog_reward'])
                #    epoch_jg.append(ev['joint_goal'])
                #    print('JG: ', ev['joint_goal'], 'BS:', ev['belief_state'], 'DR:', ev['dialog_reward'])
                # Save for train predictions for evaluation

                for i in range(len(batch)):
                    train_predictions.append((predictions[i],
                                              turn_predictions[i]))

            #print(epoch_rewards)
            #print(epoch_jg)
            # evalute on train and dev
            summary = {'iteration': iteration, 'epoch': self.epochs_trained}
            for k, v in track.items():
                summary[k] = sum(v) / len(v)
            self.logger.info("Evaluating...")
            predictions, turn_predictions = zip(*train_predictions)
            summary.update({'eval_train_{}'.format(k): v for k, v in
                            evaluate_preds(dialogs_train, predictions,
                                           turn_predictions, args.eval_domains
                                           ).items()})
            summary.update({'eval_dev_{}'.format(k): v for k, v in
                            self.run_eval(dialogs_dev, s2v, args.eval_domains,
                                          self.args.dout +
                                          "/prediction_dv_{}_{}.json".
                                          format(epoch, str(args.eval_domains))
                                          ).items()})

            global_mean_slots_filled = np.mean(global_mean_slots_filled)
            #self.logger.info("Predicted {}% slots as present".format(
            #    global_mean_slots_filled * 100))
            #self.logger.info("Epoch summary: " + str(summary))

            # do early stopping saves
            stop_key = 'eval_dev_{}'.format(args.stop)
            train_key = 'eval_train_{}'.format(args.stop)
            if best.get(stop_key, 0) <= summary[stop_key]:
                no_improvements_for = 0
                best_dev = '{:f}'.format(summary[stop_key])
                best_train = '{:f}'.format(summary[train_key])
                best.update(summary)
                self.save(best,
                          identifier='epoch={epoch},iter={iteration},'
                                     'train_{key}={train},dev_{key}={dev}'
                                     ''.
                          format(epoch=self.epochs_trained,
                                 iteration=iteration, train=best_train,
                                 dev=best_dev, key=args.stop))
                self.prune_saves()
            else:
                no_improvements_for += 1
                if no_improvements_for > args.patience:
                    self.logger.info("Ending training after model did not"
                                     "improve for {} epochs".format(
                                                no_improvements_for))
                    break

            summary.update({'best_{}'.format(k): v for k, v in best.items()})
            self.logger.info(pformat(summary))
            track.clear()

    def discount_rewards(self, rewards, gamma):
        """
            Compute discounted reward
        """
        R = 0
        rews = []
        for r in rewards[::-1]:
            R = r + gamma * R
            rews.insert(0, R)
        rews = torch.FloatTensor(rews)
        return rews

    def reinforce_update(self, batch_rewards, log_probs, gamma, base_reward, entropies):
        policy_loss = []
        beta = 0.05 # Entropy weight

        # Compute discounted rewards
        rewards = self.discount_rewards(batch_rewards, gamma)
        base_rewards = self.discount_rewards(base_reward, gamma)

        #rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        #base_rewards = (base_rewards - base_rewards.mean()) / (base_rewards.std() + eps)
        #print(rewards.mean())
        #print(base_rewards.mean())
        for log_prob, reward, base, entropy in zip(log_probs, rewards, base_rewards, entropies):
            if np.isnan(reward):
                reward = 0
            if np.isnan(base):
                base = 0
            policy_loss.append(-log_prob * (reward - base) + entropy * beta)
            #policy_loss.append(-log_prob * reward)
            # policy_loss.append(-log_prob * (reward+0.001))
            # local_policy_loss = -log_prob * (reward+0.001)
            # local_policy_loss.backward()
            # policy_loss.backward()
            # policy_loss.append(reward)
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum() #/ len(log_probs)
        #print("-- POLICY LOSS --", policy_loss, type(policy_loss))
        # policy_loss = torch.autograd.Variable(policy_loss)
        #print(policy_loss)
        # policy_loss.backward(retain_graph=True)
        try:
            policy_loss.backward()
            #for p in self.parameters():
            #    p.data.add_(-self.args.lr, p.grad.data)
            self.optimizer.step()
        except RuntimeError:
            print("WARNING! couldn't update with policy loss:", policy_loss)


    def reinforce_loss(self, bi_rewards, log_probs, base_reward, entropies, gamma):
        """
            Calculate the loss for a batch items using discounted future rewards, entropy
            and the advantage for variance reduction
        """
        if len(bi_rewards) == len(log_probs) == len(entropies) == 0:
            return torch.tensor(0., requires_grad=True).sum()

        policy_loss = []
        beta = 0.05

        # TODO Binary reward better/worse than base

        # Discounted reward
        #rewards = bi_rewards
        #base_rewards = base_reward
        rewards = self.discount_rewards(bi_rewards, gamma)
        base_rewards = self.discount_rewards(base_reward, gamma)

        # Centering
        #rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        #base_rewards = (base_rewards - base_rewards.mean()) / (base_rewards.std() + eps)

        # Compute accumulated dialog loss
        for log_prob, reward, base, entropy in zip(log_probs, rewards, base_rewards, entropies):
            if np.isnan(reward):
                reward = 0
            if np.isnan(base):
                base = 0
            # Calculate 'turn' loss
            advantage = reward - base
            policy_loss.append(-log_prob * advantage + entropy * beta)

        # Return dialog loss
        return torch.stack(policy_loss).sum()

    def reinforce_update_losses(self, batch_losses):
        """
            Update the policy from computed batch losses
        """

        self.optimizer.zero_grad()
        policy_loss = torch.stack(batch_losses).sum()

        #try:
        policy_loss.backward()
            #for p in self.parameters():
            #    p.data.add_(-self.args.lr, p.grad.data)
        self.optimizer.step()
        #except RuntimeError:
        #    print("WARNING! couldn't update with policy loss:", policy_loss)

    def run_pred(self, dialogs, s2v):
        self.eval()
        predictions_d, turn_predictions, _, _, _ = self.forward(dialogs, s2v)
        return predictions_d, turn_predictions

    def run_eval(self, dialogs, s2v, eval_domains, outfile):
        if torch.cuda.is_available() and self.device.type == 'cuda':
            s2v = util.s2v_to_device(s2v, self.device)
        predictions, turn_predictions = self.run_pred(dialogs, s2v)
        return evaluate_preds(dialogs, predictions, turn_predictions,
                              eval_domains, outfile)

    def save(self, summary, identifier):
        fname = '{}/{}.t7'.format(self.args.dout, identifier)
        logging.info('saving model to {}'.format(fname))
        state = {
            'args': vars(self.args),
            'model': self.state_dict(),
            'summary': summary,
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epochs_trained
        }
        torch.save(state, fname)

    def load_rl_model(self, revert=True):
        self.logger.info('Reverting to previous best model')
        fname = '{}/best-rl.t7'.format(self.args.dout)
        state = torch.load(fname, map_location=lambda storage, loc: storage)
        self.set_optimizer()
        resume_from_epoch = state.get('epoch', 0)
        self.set_epochs_trained(resume_from_epoch)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def load(self, path):
        self.logger.info('loading model from {}'.format(path))
        #state = torch.load(path)
        state = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state['model'])
        self.set_optimizer()
        self.optimizer.load_state_dict(state['optimizer'])
        resume_from_epoch = state.get('epoch', 0)
        self.set_epochs_trained(resume_from_epoch)
        self.logger.info("Resuming from epoch {}".format(resume_from_epoch))
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def prune_saves(self, n_keep=5):
            scores_and_files = self.get_saves()
            if len(scores_and_files) > n_keep:
                for score, fname in scores_and_files[n_keep:]:
                    os.remove(fname)

    def load_best_save(self, directory):
        if directory is None:
            directory = self.args.dout

        scores_and_files = self.get_saves(directory=directory)
        print(scores_and_files)
        if scores_and_files:
            assert scores_and_files, 'no saves exist at {}'.format(directory)
            score, fname = scores_and_files[0]
            self.load(fname)

    def get_saves(self, directory=None):
        if directory is None:
            directory = self.args.dout
        files = [f for f in os.listdir(directory) if f.endswith('.t7')]
        scores = []
        for fname in files:
            re_str = r'dev_{}=([0-9\.]+)'.format(self.args.stop)
            dev_acc = re.findall(re_str, fname)
            if dev_acc:
                score = float(dev_acc[0].strip('.'))
                scores.append((score, os.path.join(directory, fname)))
        if not scores:
            raise Exception('No files found!')
        scores.sort(key=lambda tup: tup[0], reverse=True)
        return scores

    def get_device(self, device_id):
        if device_id is not None and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu = device_id % num_gpus
            return torch.device('cuda:{}'.format(gpu))
        else:
            return torch.device('cpu')

