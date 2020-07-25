import numpy as np

f = open("nodeRules2matrix.txt", 'r')

node_rules = np.zeros((44,44))

for line in f.readlines():
    line = line.strip('\n')
    row_node, column_node = line.split(' ')
    node_rules[int(row_node)-1][int(column_node)-1] = 1

print(node_rules)