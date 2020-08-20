import networkx as nx

def extract(name, num,total_feature):

    f = open('nodeLabel.txt', 'r')
    f1 = open('nodeValue_468_50.txt', 'r')

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
    node_list_dic = dict(G[num].nodes._nodes)
    node_feature = list(next(iter(node_list_dic.values())).keys())
    print(node_list)
    print(G[num].nodes.data())
    print(type(node_list_dic))

    # Generate edge list
    edges_list = G[num].edges

    # Generate node type list
    node_type_list_num = []
    node_value_type_list =[]
    node_value = []

    for i in node_list:
        for j in node_feature:
            if 'type' in j and 'value' not in j:
                if G[num].node[i][j] == 1:
                    m, n = j.split('e')
                    node_type_list_num.append(n)
            elif 'type' in j and 'value' in j:
                if G[num].node[i][j] == 1:
                    m, n = j.split('type')
                    node_value_type_list.append(n)
            elif 'int' in j and 'value' in j:
                if G[num].node[i]['value_type0'] == 1:
                    node_value.append(int(G[num].node[i][j]))
            elif 'float' in j and 'value' in j:
                if G[num].node[i]['value_type1'] == 1:
                    node_value.append(float(G[num].node[i][j]))
            elif 'string' in j and 'value' in j:
                if G[num].node[i][j] == 1:
                    m, n = j.split('value')
                    node_value.append(n)
    # print(node_type_list_num)
    # print(node_value_type_list)
    # print(node_value)
        # for j in range(total_feature):
        #     feature = 'f' + str(j)
        #     print(feature)
        #     if G[num].node[i][feature] == 1:
        #         node_type_list_num.append(j)

    # print(node_type_list_num)
    total_type = []
    count = []
    node_type_list = []

    for line in f.readlines():
        line = line.strip('\n')
        type_name, type_ID = line.split(' ')

        total_type.append(type_name)
        count.append(type_ID)

    for i in node_type_list_num:
        p = count.index(str(int(i)+1))
        new_type = total_type[p]
        node_type_list.append(new_type)

    total_value = []
    count_v = []
    node_value_list = []
    for line in f1.readlines():
        line = line.strip('\n')
        value_id, value_name = line.split(': ')
        total_value.append(value_name)
        count_v.append(value_id)
    for i in node_value:
        if type(i) != int or type(i) != float:
            p = count_v.index(str(int(i)+1))
            new_value = total_value[p]
            node_value_list.append(new_value)
        else:
            node_value_list.append(i)
    # print(node_value_list)

    # Generate edge type list
    # TODO: wait for tthe decision of edge type
    # edge_type_list = []
    # for i in edges_list:
    #     m, n = i[0], i[1]
    #     for j in range(2):
    #         feature = 'f' + str(j+1)
    #         if G[3][m][n] == 1:
    #             edge_type_list.append(j+1)

    return node_list, edges_list, node_type_list, node_value_list, node_value_type_list