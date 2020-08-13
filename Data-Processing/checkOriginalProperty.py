import json
f = open('normalize-500-50-nor.json', 'r')
f1 = open('properties.txt', 'r')
p = []
for line in f1.readlines():
    line = line.strip('\n')
    p.append(line)
property = []
for line in f.readlines():
    dics = json.loads(line)
    for i in range(len(dics)):
        if isinstance(dics[i], dict):
            if dics[i]['type'] == 'Identifier' and 'value' in dics[i].keys() and dics[i]['value'] not in property and dics[i]['value'] not in p:
                property.append(dics[i]['value'])
print(len(property))
print(property)