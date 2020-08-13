import networkx as nx
import json
# import matplotlib.pyplot as plt

f = open('normalize-500-50-nor.json', 'r')
w1 = open('AST_A.txt', 'a')
w2 = open('AST_graph_indicator.txt', 'a')
w3 = open('AST_graph_labels.txt', 'a')
w4 = open('AST_node_labels_1.txt', 'a')
# w5 = open('500-50-norequires.json', 'a')
w6 = open('AST_nodel_value.txt', 'a')

number_of_nodes = 0
total_nodes = 1
total_edges = 0
flag = 0
count = 0
min_node = 45
max_node = 55

for line in f.readlines():
    dics = json.loads(line)
    if len(dics) < max_node+2 and len(dics) > min_node +2:
    # if len(dics)<min_node:
    #
        if flag < 500:
            curr = []
            check = []
            for i in range(len(dics)):
                if isinstance(dics[i], dict):
                    if dics[i]['type'] == 'Property':
                        if 'value' in dics[i].keys():
                            curr.append((dics[i]['value']))
            # if len(curr)>25 and len(dics)<max_node+2 and flag<50:
            if 'require' not in curr:
                flag += 1
                # w5.write(line)
                G = nx.Graph()

                for i in range(len(dics)):
                    if isinstance(dics[i], dict):
                        if G.has_node(dics[i]['id']+total_nodes) == False:
                            G.add_node(dics[i]['id'] + total_nodes, name = dics[i]['type'])

                            w4.write(dics[i]['type'])
                            w4.write('\n')

                            if 'value' in dics[i].keys():
                                node_value = dics[i]['type'] + ': ' + str(dics[i]['value'])
                                w6.write(node_value)
                                w6.write('\n')
                            else:
                                node_value = dics[i]['type'] + ': ' + 'NoValue'
                                w6.write(node_value)
                                w6.write('\n')

                            if 'children' in dics[i].keys():
                                curr_node = [x + total_nodes for x in dics[i]['children']]
                                G.add_nodes_from(curr_node)
                                for j in curr_node:
                                    G.add_edge(dics[i]['id'] + total_nodes,j)

                        else:
                            G.node[dics[i]['id']+total_nodes]['name'] = dics[i]['type']

                            w4.write(dics[i]['type'])
                            w4.write('\n')

                            if 'value' in dics[i].keys():
                                node_value = dics[i]['type'] + ': ' + str(dics[i]['value'])
                                w6.write(node_value)
                                w6.write('\n')
                            else:
                                node_value = dics[i]['type'] + ': ' + 'NoValue'
                                w6.write(node_value)
                                w6.write('\n')

                            if 'children' in dics[i].keys():
                                curr_node = [x + total_nodes for x in dics[i]['children']]
                                G.add_nodes_from(curr_node)
                                for j in curr_node:
                                    G.add_edge(dics[i]['id'] + total_nodes,j)

                number_of_nodes = G.number_of_nodes()
                number_of_edges = G.number_of_edges()

                for i in range(total_nodes, total_nodes+number_of_nodes):

                    cur = list(G.adj[i])

                    for j in cur:
                        w1.write('%s, %d' % (j, i))
                        w1.write('\n')

                total_nodes += number_of_nodes
                total_edges += number_of_edges

                for i in range(0, number_of_nodes):
                    w2.write(str(flag))
                    w2.write('\n')

for i in range(0, flag):
    w3.write('1')
    w3.write('\n')

print('n: %d' %(total_nodes-1))
print('m: %d' %(total_edges))
print('N: %d' %(flag))



    # print(G.nodes.data())
    # print(list(nx.bfs_tree(G,0)))
    # nx.draw(G)
    # plt.savefig('test.png', bbox_inches='tight')
    # plt.show()


