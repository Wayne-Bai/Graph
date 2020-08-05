import networkx as nx
import graphviz

def extract(G, total_feature):

    # f = open('/home/zfk/Documents/graph-generation/debug/Graph/Visualization/nodeLabel.txt', 'r')
    #
    # G = nx.read_gpickle(name)

    # print(len(G))
    #
    # for i in range(len(G)):
    #     print('Graph %d'%i, G[i].number_of_nodes(), G[i].number_of_edges())

    # for i in range(3):
    #     pos = nx.spring_layout(G[i], scale=0.5)
    #     nx.draw(G[i], pos, node_size=10, with_labels=False)
    #     plt.savefig('fig%d.png'%i, bbox_inches='tight')
    #     plt.show()

    # Generate node list
    node_list = G.nodes
    print(node_list)
    print(G.nodes.data())

    # Generate edge list
    edges_list = G.edges

    # Generate node type list
    node_type_list = []

    # print(G[num].nodes[0]['f0'])
    for i in node_list:
        # for j in range(total_feature):
        #     feature = 'f' + str(j)
        #     print(feature)
        #     if G.nodes[i][feature] == 1:
        #         node_type_list_num.append(j)
        node_type_list.append(G.nodes[i]['name'])
    # print(node_type_list_num)
    total_type = []
    count = []
    # node_type_list = []

    # for line in f.readlines():
    #     line = line.strip('\n')
    #     type_name, type_ID = line.split(' ')
    #
    #     total_type.append(type_name)
    #     count.append(type_ID)
    # print(count)
    #
    # for i in node_type_list_num:
    #     p = count.index(str(i+1))
    #     new_type = total_type[p]
    #     node_type_list.append(new_type)

    # Generate edge type list
    # TODO: wait for tthe decision of edge type
    # edge_type_list = []
    # for i in edges_list:
    #     m, n = i[0], i[1]
    #     for j in range(2):
    #         feature = 'f' + str(j+1)
    #         if G[3][m][n] == 1:
    #             edge_type_list.append(j+1)

    return node_list, edges_list, node_type_list

def Visualize(G, num, total_feature):
    # pre_path = '/home/zfk/Documents/graph-generation/debug/Graph/' + path
    # path_whole = pre_path + '/graphs/'
    # if kind == 0:
    #     prefix = 'GraphRNN_RNN_AST_4_128_pred_%d_1'%(name)
    # else:
    #     prefix = 'GraphRNN_RNN_AST%d_4_128_pred_%d_1'%(kind, name)
    # print(prefix)
    # format = '.dat'
    # file = path_whole+prefix+format
    # file = prefix+str(name)+format
    if num == None:
        for c in range(16):
            node_list, edge_list, node_type_list = extract(G, total_feature)

            dot = graphviz.Graph(comment='Result', format='png')

            # Add node
            flag = 0
            print(len(node_list))
            print(len(node_type_list))
            for i in node_list:
                node_id = i
                node = node_type_list[flag]
                print(node_id, node)
                node_p = '[{}]'.format((node_id)) + ': ' + node
                dot.node(str(node_id), node_p)
                flag += 1

            for i in edge_list:
                m, n = i[0], i[1]
                dot.edge(str(m), str(n))

            dot.render("process/%s.gv" % (str(num)), view=True)

    else:
        node_list, edge_list, node_type_list = extract(G, total_feature)

        dot = graphviz.Graph(comment='Result',format='png')

        # Add node
        flag = 0
        print(len(node_list))
        print(len(node_type_list))
        for i in node_list:
            node_id = i
            node = node_type_list[flag]
            node_p = '[{}]'.format(node_id) + ': ' + node
            # print(node_id, node)
            dot.node(str(node_id), node_p)
            flag += 1

        for i in edge_list:
            m, n = i[0], i[1]
            dot.edge(str(m), str(n))

        dot.render("process/%s.gv" %(str(num)), view=True)

if __name__ == '__main__':

    # # #TODO: this section depends on choice. (Generate all processes or only the final result)
    # #
    # # for i in range(3000, 5050, 500):
    # #    Visualize('Code-0730-v3', 50, i, 0, 28)
    # Visualize('Code-0730', 2, 1800, 0, 23)

    import networkx as nx
    import json

    # import matplotlib.pyplot as plt

    f = open('programs_training_2.json', 'r')
    # w1 = open('AST_A.txt', 'a')
    # w2 = open('AST_graph_indicator.txt', 'a')
    # w3 = open('AST_graph_labels.txt', 'a')
    # w4 = open('AST_node_labels_1.txt', 'a')
    # w5 = open('more35deatures_300.json', 'a')

    number_of_nodes = 0
    total_nodes = 1
    total_edges = 0
    flag = 0
    count = 0
    # min_node = 8
    max_node = 200

    for line in f.readlines():
        dics = json.loads(line)
        # if len(dics) > min_node+2 and len(dics) < max_node +2:
        # if len(dics)<min_node:
        #
        #     if flag < 2:
        curr = []
        for i in range(len(dics)):
            if isinstance(dics[i], dict):
                if dics[i]['type'] not in curr:
                    curr.append((dics[i]['type']))
        # if len(curr) > 25 and len(dics) < max_node + 2 and flag < 50:
        flag += 1
            # w5.write(line)
        G = nx.Graph()

        for i in range(len(dics)):
            if isinstance(dics[i], dict):
                if G.has_node(dics[i]['id']+total_nodes) == False:
                    G.add_node(dics[i]['id'] + total_nodes, name=dics[i]['type'])

                    # w4.write(dics[i]['type'])
                    # w4.write('\n')

                    if 'children' in dics[i].keys():
                        curr_node = [x + total_nodes for x in dics[i]['children']]
                        G.add_nodes_from(curr_node)
                        for j in curr_node:
                            G.add_edge(dics[i]['id'] + total_nodes, j)

                else:
                    G.node[dics[i]['id']+total_nodes]['name'] = dics[i]['type']

                    # w4.write(dics[i]['type'])
                    # w4.write('\n')

                    if 'children' in dics[i].keys():
                        curr_node = [x + total_nodes for x in dics[i]['children']]
                        G.add_nodes_from(curr_node)
                        for j in curr_node:
                            G.add_edge(dics[i]['id'] + total_nodes, j)
        print(G.nodes(data=True))
        number_of_nodes = G.number_of_nodes()
        number_of_edges = G.number_of_edges()

        for i in range(total_nodes, total_nodes + number_of_nodes):

            cur = list(G.adj[i])

            # for j in cur:
            #     w1.write('%s, %d' % (j, i))
            #     w1.write('\n')

        total_nodes += number_of_nodes
        total_edges += number_of_edges

        Visualize(G,flag,23)

            # for i in range(0, number_of_nodes):
            #     w2.write(str(flag))
            #     w2.write('\n')

    # for i in range(0, flag):
    #     w3.write('1')
    #     w3.write('\n')

    print('n: %d' % (total_nodes - 1))
    print('m: %d' % (total_edges))
    print('N: %d' % (flag))


    # print(G.nodes.data())
    # print(list(nx.bfs_tree(G,0)))
    # nx.draw(G)
    # plt.savefig('test.png', bbox_inches='tight')
    # plt.show()


