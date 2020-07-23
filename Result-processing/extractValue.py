import json

f = open("programs_eval.json", 'r')
w1 = open("nodeValue.txt", 'a')

value_dic = {}

for line in f.readlines():
    dics = json.loads(line)
    for i in range(len(dics)):
        if isinstance(dics[i], dict):
            if 'value' in dics[i].keys():
                if dics[i]['type'] not in value_dic.keys():
                    value_dic[dics[i]['type']] = [dics[i]['value']]
                else:
                    value_dic[dics[i]['type']].append(dics[i]['value'])

for k,v in value_dic.items():
    flag = 1
    for c in list(set(v)):
        w = open(k+".txt", "a")
        w.write(str(flag) + "： " + str(c))
        w.write('\n')

        w1.write(str(k) + ': ' + str(c))
        w1.write('\n')
        flag += 1





