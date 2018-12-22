from collections import namedtuple

Turn = namedtuple("Turn", ["user_utt", "system_act", "system_utt",
                           "x_utt", "x_act", "x_sys", "labels", "labels_str",
                           "belief_state"])
Dialog = namedtuple("Dialog", ["turns"])
Slot = namedtuple("Slot", ["domain", "embedding", "values"])
Value = namedtuple("Value", ["value", "embedding", "idx"])
