import json

f = open("programs_eval.json", 'r')
w = open("pluginProperty.txt", 'a')

all_properties = []
properties = []

for line in f.readlines():
    dics = json.loads(line)
    for i in range(len(dics)):
        curr = []
        if isinstance(dics[i], dict):
            if dics[i]['type'] == 'Property':
                if 'value' in dics[i].keys():
                    if dics[i]['value'] not in curr:
                        curr.append((dics[i]['value']))
                    if dics[i]['value'] in all_properties and dics[i]['value'] not in properties:
                        w.write(str(dics[i]['value']))
                        w.write('\n')
                        properties.append(dics[i]['value'])
                        # all_properties.append(dics[i]['value'])
                    else:
                        all_properties.append(dics[i]['value'])
        all_properties += curr

# for property in properties:
#     w.write(property)
#     w.write('\n')