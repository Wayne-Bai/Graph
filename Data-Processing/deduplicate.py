import json

f = open('dataset_54graphs/programs_training_20nodes.json', 'r')
w = open('deduplicate_54graphs.json', 'a')

len_list = []
last_id = []
flag = 0

for line in f.readlines():
    dics = json.loads(line)
    if len(str(dics)) not in len_list:
        len_list.append(len(str(dics)))
        last_id.append(dics[-2]['id'])
        w.write(line)
    else:
        if dics[-2]['id'] != last_id[flag]:
            last_id.append((dics[-2]['id']))
            w.write(line)