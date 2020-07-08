import graphviz
import extractInfo

node_list, edge_list, node_type_list= extractInfo.extract('GraphRNN_RNN_AST_4_128_train_0.dat')

dot = graphviz.Graph(comment='Result')

# Add node
flag = 0
for i in node_list:
    node_id = i
    node = node_type_list[flag]
    print(node_id, node)
    dot.node(str(node_id), node)
    flag += 1

for i in edge_list:
    m, n = i[0], i[1]
    dot.edge(str(m), str(n))

# dot.edges(['AB','AC','AB'])
# dot.edge('B', 'C', 'test')

print(dot.source)
dot.render('test-output/test.gv', view=True)
