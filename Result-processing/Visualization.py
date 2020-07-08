import graphviz
import networkx as nx
import extractInfo

node_list, edge_list, node_type_list= extractInfo.extract('GraphRNN_RNN_AST_4_128_train_0.dat')
print(node_list)
print(edge_list)
print(node_type_list)

# #TESTq
# dot = graphviz.Digraph(comment='Test')
#
# dot.node('A', 'Dot A')
# dot.node('B', 'Dot B')
# dot.node('C', 'Dot C')
#
# dot.edges(['AB','AC','AB'])
# dot.edge('B', 'C', 'test')
#
# print(dot.source)
# dot.render('test-output/test-table.gv', view=True)
