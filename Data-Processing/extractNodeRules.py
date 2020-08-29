import json

f1 = open('programs_eval.json', 'r')
f2 = open('nodeLabel.txt', 'r')

w1 = open('nodeRules.txt', 'a')

node_dic = {}
total_type = []
count = []
key_list = []
new_key_list = []

for line in f1.readlines():
    dics = json.loads(line)
    for i in range(len(dics)):
        if isinstance(dics[i], dict):
            if 'children' in dics[i].keys():
                for j in dics[i]['children']:
                    if dics[i]['type'] not in node_dic.keys():
                        node_dic[dics[i]['type']] = [dics[j]['type']]
                    else:
                        if dics[j]['type'] not in node_dic[dics[i]['type']]:
                            node_dic[dics[i]['type']].append(dics[j]['type'])

for k,v in node_dic.items():
    w1.write(k + ":")
    for i in v:
        w1.write(' ' + i)
    w1.write('\n')

# for line in f2.readlines():
#     line = line.strip('\n')
#     type_name, type_ID = line.split(' ')
#
#     total_type.append(type_name)
#     count.append(type_ID)
#
# for line in f1.readlines():
#     dics = json.loads(line)
#     for i in range(len(dics)):
#         if isinstance(dics[i], dict):
#             if 'children' in dics[i].keys():
#                 p = total_type.index(dics[i]['type'])
#                 new_k = count[p]
#                 for j in dics[i]['children']:
#                     p1 = total_type.index(dics[j]['type'])
#                     new_value = count[p1]
#                     if new_k not in node_dic.keys():
#                         node_dic[new_k] = [new_value]
#                     else:
#                         if new_value not in node_dic[new_k]:
#                             node_dic[new_k].append(new_value)

print(node_dic)