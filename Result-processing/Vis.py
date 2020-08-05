import networkx as nx
import graphviz

def extract(name, num,total_feature):

    f = open('/home/zfk/Documents/graph-generation/debug/Graph/Visualization/nodeLabel.txt', 'r')

    G = nx.read_gpickle(name)

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
    node_list = G[num].nodes
    print(node_list)
    print(G[num].nodes.data())

    # Generate edge list
    edges_list = G[num].edges

    # Generate node type list
    node_type_list_num = []

    # print(G[num].nodes[0]['f0'])
    for i in node_list:
        for j in range(total_feature):
            feature = 'f' + str(j)
            print(feature)
            if G[num].nodes[i][feature] == 1:
                node_type_list_num.append(j)

    print(node_type_list_num)
    total_type = []
    count = []
    node_type_list = []

    for line in f.readlines():
        line = line.strip('\n')
        type_name, type_ID = line.split(' ')

        total_type.append(type_name)
        count.append(type_ID)
    print(count)

    for i in node_type_list_num:
        p = count.index(str(i+1))
        new_type = total_type[p]
        node_type_list.append(new_type)

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

def Visualize(path, kind, name, num, total_feature):
    pre_path = '/home/zfk/Documents/graph-generation/debug/Graph/' + path
    path_whole = pre_path + '/graphs/'
    if kind == 0:
        prefix = 'GraphRNN_RNN_AST_4_128_pred_%d_1'%(name)
    else:
        prefix = 'GraphRNN_RNN_AST%d_4_128_pred_%d_1'%(kind, name)
    print(prefix)
    format = '.dat'
    file = path_whole+prefix+format
    # file = prefix+str(name)+format
    if num == None:
        for c in range(16):
            node_list, edge_list, node_type_list = extract(file, c, total_feature)

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

            dot.render("process/%s/%s.gv" % (path, str(name)+'-'+str(c)), view=True)

    else:
        node_list, edge_list, node_type_list = extract(file, num, total_feature)

        dot = graphviz.Graph(comment='Result',format='png')

        # Add node
        flag = 0
        print(len(node_list))
        print(len(node_type_list))
        for i in node_list:
            node_id = i
            node = node_type_list[flag]
            print(node_id, node)
            dot.node(str(node_id), node)
            flag += 1

        for i in edge_list:
            m, n = i[0], i[1]
            dot.edge(str(m), str(n))

        dot.render("process/%s/%s.gv" %(path,name), view=True)

if __name__ == '__main__':

    #TODO: this section depends on choice. (Generate all processes or only the final result)

    for i in range(3000, 5050, 500):
       Visualize('Code-0730-v3', 50, i, 0, 28)
    # Visualize('Code-0730', 2, 1800, 0, 23)