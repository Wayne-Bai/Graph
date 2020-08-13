f = open('AST_node_value_1.txt', 'r')
w1 = open('nodeValue_500_50.txt', 'a')

property = []
unnum_property = []

for line in f.readlines():
    line = line.strip('\n')
    id, ty, value = line.split(': ')

    if ty == 'LiteralNumber' and '.' in value:
        value = float(value)
    elif ty == 'LiteralNumber' and '.' not in value:
        value = int(value)
    elif ty == 'Identifier' and '.' in value:
        value = float(value)
    elif ty == 'Property' and '.' in value:
        value = float(value)
    elif ty == 'Indetifier' or type == 'Property':
        try:
            value = int(value)
        except ValueError:
            pass

    if value not in property:
        property.append(value)
    if type(value) != int and type(value) != float and value not in unnum_property:
        unnum_property.append(value)

print(len(property))
print(property)
print(len(unnum_property))
print(unnum_property)
print(sorted(unnum_property))

sn = sorted(unnum_property)
print(len(sn))
sn.remove('NoValue')
sn.insert(0, 'NoValue')
print(sn)
print(len(sn))

# for i in range(len(unnum_property)):
#     w1.write(str(i+1) + ': ' + str(unnum_property[i]))
#     w1.write('\n')
# #
# for i in range(len(property)):
#     w2.write(str(i+1) + ': ' + str(property[i]))
#     w2.write('\n')
for i in range(len(sn)):
    w1.write(str(i+1) + ': ' + str(sn[i]))
    w1.write('\n')
# dic = {}
#
# for k,v in enumerate(sorted(unnum_property)):
#     node_id = str(k) + ": "
#     w1.write(node_id)
#     w1.write(v)
#     w1.write('\n')
#
# for k,v in enumerate(sorted(property)):
#     node_id = str(k) + ": "
#     w2.write(node_id)
#     w2.write(v)
#     w2.write('\n')