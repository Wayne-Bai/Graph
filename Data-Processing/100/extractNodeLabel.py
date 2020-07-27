f = open('AST_node_labels_1.txt', 'r')
w1 = open('nodeLabel.txt', 'r')
w2 = open('AST_node_labels.txt', 'a')

total = []
total_type = []
count = []
# Extract node label
for line in f.readlines():
    line = line.strip('\n')

    total.append(line)

# count = [i for i in range(1, len(total_type)+2)]
for line in w1.readlines():
    line = line.strip('\n')
    type_name, type_ID = line.split(' ')

    total_type.append(type_name)
    count.append(type_ID)

# print(total_type)
# print(count)


# for i in range(len(total_type)):
#     w2.write('%s: %d' %(total_type[i], count[i]))
#     w2.write('\n')

#
for i in total:
    p = total_type.index(i)
    new_type = count[p]
    w2.write(str(new_type))
    w2.write('\n')