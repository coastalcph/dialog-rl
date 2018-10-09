from collections import namedtuple
import random
import torch
import numpy as np

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
#
# # if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# optimizer = torch.optim.RMSprop(policy_net.parameters())
#
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

#
# class ReplayMemory(object):
#
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0
#
#     def push(self, *args):
#         """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)
#
#
# def optimize_model(memory):
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)
#     # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation).
#     batch = Transition(*zip(*transitions))
#
#     # Compute a mask of non-final states and concatenate the batch elements
#     non_final_mask = torch.tensor(tuple(map(lambda s:s is not None,
#                                             batch.next_state)),
#                                   device=device, dtype=torch.uint8)
#     non_final_next_states = torch.cat([s for s in batch.next_state
#                                        if s is not None])
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat(batch.reward)
#
#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken
#     state_action_values = policy_net(state_batch).gather(1, action_batch)
#
#     # Compute V(s_{t+1}) for all next states.
#     next_state_values = torch.zeros(BATCH_SIZE, device=device)
#     next_state_values[non_final_mask] = \
#     target_net(non_final_next_states).max(1)[0].detach()
#     # Compute the expected Q values
#     expected_state_action_values = (
#                                                next_state_values * GAMMA) + reward_batch
#
#     # Compute Huber loss
#     loss = F.smooth_l1_loss(state_action_values,
#                             expected_state_action_values.unsqueeze(1))
#
#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     for param in policy_net.parameters():
#         param.grad.data.clamp_(-1, 1)
#     optimizer.step()
#

def get_rewards(data, preds):
    request = []
    inform = []
    joint_goal = []
    fix = {'centre':'center', 'areas':'area', 'phone number':'number'}
    i = 0
    for dialog in data:
        pred_state = {}
        for t in dialog.turns:
            gold_request = set(
                [(s, v) for s, v in t.turn_label if s == 'request'])
            gold_inform = set(
                [(s, v) for s, v in t.turn_label if s != 'request'])
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
                        gold_recovered.add((b['act'],
                                            fix.get(s.strip(), s.strip()),
                                            fix.get(v.strip(), v.strip())))
            for s, v in pred_state.items():
                pred_recovered.add(('inform', s, v))
            joint_goal.append(gold_recovered == pred_recovered)
            i += 1

    return {'turn_inform':np.mean(inform), 'turn_request':np.mean(request),
            'joint_goal':np.mean(joint_goal)}