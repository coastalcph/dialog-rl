import numpy as np

def evaluate_preds(dialogs, preds, turn_predictions):

    def equals_lower(first, second):
        first = (v.lower() for v in first)
        second = (v.lower() for v in second)
        return first == second

    # print("PREDS", preds, len(preds))
    # print("TURN PREDS", turn_predictions, len(turn_predictions))

    request = []
    inform = []
    joint_goal = []
    fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
    for di, d in enumerate(dialogs):
        pred_state = {}
        dialog_gold = d[-1]

        for ti, turn in enumerate(d.turns):
            turn_gold = turn.labels
            # print("TURN GOLD: ", turn_gold)

            gold_request = {s: int(np.argmax(v)) for s, v in turn_gold.items() if s == 'request'}
            gold_inform = {s: int(np.argmax(v)) for s, v in turn_gold.items() if s != 'request'}
            pred_request = {s: v for s, v in turn_predictions[di][ti].items() if s == 'request'}
            pred_inform = {s: v for s, v in turn_predictions[di][ti].items() if s != 'request'}
            for s, v in gold_request.items():
                request.append(v == pred_request[s])

            for s, v in gold_inform.items():
                inform.append(v == pred_inform[s])

            gold_recovered = set()
            pred_recovered = set()

            for s, v in pred_inform.items():
                pred_state[s] = v

            for b in turn.belief_state:
                if b['act'] != 'request':
                    gold_recovered.add((b['act'], b['slots'][0], b['slots'][1]))
            for s, v in pred_state.items():
                pred_recovered.add(('inform', s, v))

            joint_goal.append(equals_lower(gold_recovered, pred_recovered))

    return {'turn_inform': np.mean(inform), 'turn_request': np.mean(request),
            'joint_goal': np.mean(joint_goal)}
