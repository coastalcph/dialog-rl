# coding: utf-8
import json
from pprint import pprint


with open('data.json') as f:
    data = json.load(f)

with open('ontology.json') as f:
    ont = json.load(f)
    
with open('dialogue_acts.json') as f:
    acts = json.load(f)
    
with open('testListFile.json') as f:
    test = [line.strip() for line in f.readlines()]

with open('valListFile.json') as f:
    dev = [line.strip() for line in f.readlines()]
    

def get_past_acts(belief, past_acts):

    for i in range(len( belief)):
        cur = belief[i]
        if str(cur['slots']) in past_acts.keys(): 
          
            belief[i] = {'slots': cur['slots'], 'act': past_acts[str(cur['slots'])]}
        
    return belief

def get_domain(entry):
    domains = []
    for key, value in entry['goal'].items():
        if not entry['goal'][key]:
            pass
        elif key != 'message' and key!= 'topic':
            domains.append(key) 
    return domains

def get_label(dictionary, domain):
    cur_turn_label = []
    for key2, value in dictionary.items():
                if value != 'not mentioned' and value != '':
                    cur_turn_label.append([str(domain)+'-'+str(key2), value])
    return cur_turn_label
            
def get_belief_state(entry, domains, k, acts, past_acts ):
  
    
    turn_label = []
    belief_state = []
    #cur_turn_label = []
    turn_label_update = []
    for domain in domains:
       
        semi = entry['log'][k]['metadata'][domain]['semi']
        book = entry['log'][k]['metadata'][domain]['book']
        
        if k != 1 and k != 0 :
            
            prev_semi = entry['log'][k-2]['metadata'][domain]['semi']
            prev_book = entry['log'][k-2]['metadata'][domain]['book']
            
            cur_turn_label1 = get_label(semi, domain)
            turn_label1 = get_label(prev_semi, domain)
            
            cur_turn_label2 = get_label(book, domain)
            turn_label2 = get_label(prev_book, domain)
            
            cur_turn_label = cur_turn_label1 + cur_turn_label2
            turn_label = turn_label1 + turn_label2
            
            
            for i in range(len(cur_turn_label)):
                if cur_turn_label[i] not in turn_label:
                    turn_label_update.append(cur_turn_label[i])
            
        else:
            cur_turn_label2 = get_label(book, domain)
            cur_turn_label1 = get_label(semi, domain)
            cur_turn_label = cur_turn_label1 + cur_turn_label2
                    
            prev_turn_label = []
            for i in range(len(cur_turn_label)):
                if cur_turn_label[i] not in prev_turn_label:
                    turn_label_update.append(cur_turn_label[i])

        for slot_pair in cur_turn_label:
            if len(slot_pair[1]) != 0 and type(slot_pair[1][0]) == dict:
                d = slot_pair[1][0]
                
                slots_new = []
                for key, value in d.items():
                   
                    slots_new.append([str(domain)+'-'+str(key),value])
                #k is the turn number      
                for p in slots_new:
                    belief_state.append({'slots': p, 'act': ""})
            else:  
                belief_state.append({'slots': slot_pair, 'act':''})
                    
    return belief_state, turn_label_update, past_acts


def check_dict(sample):
    instances = [isinstance(x, dict) for x in sample]
    return True in instances

def clean_data(data):
    
    for i in range(len(data)):
        domains = data[i]['domain']
        look = [str(domain)+"-"+'booked'for domain in domains]
                    
        for turn in data[i]['dialogue']:
            new_belief = []
            if len(turn['belief_state']) != 0:
                for m in look:
                    for j  in range(len(turn['belief_state'])):
                    
                        if 'booked' not in turn['belief_state'][j]['slots'][0] and turn['belief_state'][j]['slots'][1] != []:
                            if {'slots': turn['belief_state'][j]['slots'], "act": ""} not in new_belief:

                                new_belief.append({'slots': turn['belief_state'][j]['slots'], "act": ""})
             
                        if [m, []] in turn['turn_label']:
                            #print([m, []] )
                            turn['turn_label'].remove([m, []])
            if len(turn['turn_label']) > 0:
                
                for label in turn['turn_label']:
                    if label[0]== 'restaurant-booked':
                        
                        pass
                    if 'booked' in label[0]:
                       
                        dom = label[0].replace('booked',"")
                        cur = label
                        turn['turn_label'].remove(label) 
                        
                    #print(label[1], len(label[1]))
                        if check_dict(label[1]) == True:
                            #print(label[1][0])
                            for z in range(len(label[1])):
                                for key, value in label[1][z].items():
                                    turn['turn_label'].append([dom+str(key),value])

                              
                            

            turn['belief_state'] = new_belief
    return data

def act_list(cur_act):
    cur_dialogue = []
    for item in list(cur_act.values()):
        if type(item) != str:

            for key in item.keys():

                for i in item[key]:
                    cur_item = i[0].lower()
                    if cur_item != 'none':
                        cur_dialogue.append({i[0]:(key,i[1])})
    return cur_dialogue
def save_data(split, _set):
    with open(split+'.json', 'w') as fp:
        json.dump(_set, fp, sort_keys=True, indent=4)




idx = [i for i in range(len(data.keys()))]
all_files =  [key for key in data.keys()]
id2dial = dict(zip(idx, all_files))
dial2id = dict(zip( all_files, idx))



dict_list = []
past_acts = {}
for i in idx:
    dial_dict = {}
    dial_dict['dialogue_idx']= i
    dial_dict['goal']= data[id2dial[i]]['goal']
    domains = get_domain(data[id2dial[i]]) 
    dial_dict['domain'] = domains
    dial_dict['dialogue'] = []
    current_dict = {}
    turn_idx = 0
    
    for k in range(len(data[id2dial[i]]['log'])):
        if k == 0:    #on the first utterance the system transcript is always empty
            current_dict['transcript'] = data[id2dial[i]]['log'][k]['text']
            current_dict['turn_idx'] = turn_idx
            belief_state, turn_labels,  past_acts = get_belief_state(data[id2dial[i]],domains, k+1 , acts[id2dial[i].split('.')[0]], past_acts)
            current_dict['belief_state'] = belief_state
            current_dict['turn_label'] = turn_labels
            current_dict['system_transcript'] = ""
            current_dict["system_acts"] = []
    
        if bool(data[id2dial[i]]['log'][k]['metadata']) == False:     #if it is false, this is a user utterance
            current_dict['transcript'] = data[id2dial[i]]['log'][k]['text']
            try:
                
                belief_state, turn_labels, past_acts = get_belief_state(data[id2dial[i]],domains, k+1, acts[id2dial[i].split('.')[0]], past_acts )
                
                current_dict['belief_state'] = belief_state
                current_dict['turn_label'] = turn_labels
            except Exception:
                print(i)
            current_dict['turn_idx'] = turn_idx
            dial_dict['dialogue'].append(current_dict)
            current_dict =  {}  #initialize dictionary after turn has ended
            # a turn consists of system utterance and user utterance + other meta data
       
        else: 
            current_dict['system_transcript'] = data[id2dial[i]]['log'][k]['text']
            current_dict["system_acts"] = []
            try:
                belief_state, turn_labels,  past_acts = get_belief_state(data[id2dial[i]],domains, k , acts[id2dial[i].split('.')[0]], past_acts)
                current_dict['belief_state'] = belief_state
                current_dict['turn_label'] = turn_labels
            except Exception:
                pass
            turn_idx += 1    
            if k == len(data[id2dial[i]]['log']) -1:
          
            
                current_dict['turn_idx'] = turn_idx
                current_dict['transcript'] = ""
                dial_dict['dialogue'].append(current_dict)

    dict_list.append(dial_dict)


for i in range(3):
    cleaned_list = clean_data(dict_list)
###########################################################
#################      TURN LABELS     ####################
###########################################################

all_acts = []
for index in idx: 
    cur_dialogue = act_list(acts[id2dial[index].split('.')[0]])
    for turn in acts[id2dial[index].split('.')[0]].keys():
        if type(acts[id2dial[index].split('.')[0]][turn]) == str:
            pass
        else:
            try:
                for key in acts[id2dial[index].split('.')[0]][turn].keys():
                    for alist in acts[id2dial[index].split('.')[0]][turn][key]:
                        new = [key, alist[0], alist[1]]

                        cleaned_list[index]['dialogue'][int(turn)]['turn_label'].append(new)
                        for j in range(len(cleaned_list[index]['dialogue'][int(turn)]['turn_label'])):
                            if len(cleaned_list[index]['dialogue'][int(turn)]['turn_label'][j]) == 2:
                                 cleaned_list[index]['dialogue'][int(turn)]['turn_label'][j] = ['Inform', cleaned_list[index]['dialogue'][int(turn)]['turn_label'][j][0], cleaned_list[index]['dialogue'][int(turn)]['turn_label'][j][1]]
            except Exception:
                pass

    
        all_acts.append(cur_dialogue)
                

###########################################################
#################      BELIEF STATE    ####################
###########################################################

for i in idx:
    for cur in all_acts[i]:
        for key in cur.keys():
           
            for turn_idx in range(len(cleaned_list[i]['dialogue'])):
                lower = key.lower()
                for j in range(len(cleaned_list[i]['dialogue'][turn_idx]['belief_state'])):
                    slot = cleaned_list[i]['dialogue'][turn_idx]['belief_state'][j]['slots'][0]
                    val = cleaned_list[i]['dialogue'][turn_idx]['belief_state'][j]['slots'][1]
                    c = cleaned_list[i]['dialogue'][turn_idx]['belief_state'][j]['act']
                    if lower in slot  or cur[key][1] == val.lower():
                        if c == "":
                            cleaned_list[i]['dialogue'][turn_idx]['belief_state'][j]['act'] = cur[key][0]
    for turn_idx in range(len(cleaned_list[i]['dialogue'])):
        for j in range(len(cleaned_list[i]['dialogue'][turn_idx]['belief_state'])):
            c = cleaned_list[i]['dialogue'][turn_idx]['belief_state'][j]['act']
            if c== "":
                dom = slot.split('-')[0][0].upper()+slot.split('-')[0][1::]

                cleaned_list[i]['dialogue'][turn_idx]['belief_state'][j]['act'] = dom+'-Inform'



for i in idx:
    for key, value in acts[id2dial[i].split('.')[0]].items():
        cur_dialogue = cleaned_list[i]['dialogue']
        
        #turn['system_acts'].append()
        if type(value) != str:
           
            try:
                turn = cur_dialogue[int(key)]
                for k, v in value.items():
                    for item in v:
                        if item != ['none', 'none']:
                            cleaned_list[i]['dialogue'][int(key)]['system_acts'].append(item)
                
            except Exception:
                pass
  

test_ids = [dial2id[f] for f in test]
dev_ids = [dial2id[f] for f in dev]
train_ids = list(set(set(idx) - set(test_ids))-set(dev_ids))


test_set = [cleaned_list[i] for i in test_ids]
dev_set = [cleaned_list[i] for i in dev_ids]
train_set = [cleaned_list[i] for i in train_ids]




save_data('train', train_set)
save_data('test', test_set)
save_data('dev', dev_set)

    

