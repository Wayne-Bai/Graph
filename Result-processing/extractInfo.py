import networkx as nx

def extract(name, num,total_feature):

    f = open('nodeLabel.txt', 'r')

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
    node_list = G[num].node
    print(node_list)
    print(G[num].nodes.data())

    # Generate edge list
    edges_list = G[num].edges

    # Generate node type list
    node_type_list_num = []

    for i in node_list:
        for j in range(total_feature):
            feature = 'f' + str(j+1)
            if G[num].node[i][feature] == 1:
                node_type_list_num.append(j+1)

    total_type = []
    count = []
    node_type_list = []

    for line in f.readlines():
        line = line.strip('\n')
        type_name, type_ID = line.split(' ')

        total_type.append(type_name)
        count.append(type_ID)

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