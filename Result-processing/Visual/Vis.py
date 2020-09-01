import networkx as nx
import graphviz

def extract(name, num, dataset):

    f = open('/home/zfk/Documents/graph-generation/debug/Graph/Visualization/nodeLabel.txt', 'r')
    
    if dataset == '2':
        f1 = open('/home/zfk/Documents/graph-generation/debug/Graph/Visualization/nodeValue_2.txt', 'r')
    if dataset == '500-50':
        f1 = open('/home/zfk/Documents/graph-generation/debug/Graph/Visualization/nodeValue_500_50.txt', 'r')
    if dataset == '50-20':
        f1 = open('/home/zfk/Documents/graph-generation/debug/Graph/Visualization/nodeValue_50_20.txt', 'r')
    if dataset == '468-50':
        f1 = open('/home/zfk/Documents/graph-generation/debug/Graph/Visualization/nodeValue_468_50.txt', 'r')

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

    node_list = G[num].nodes
    node_list_dic = dict(G[num].nodes._nodes)
    node_feature = list(next(iter(node_list_dic.values())).keys())
    print(node_list)
    print(G[num].nodes.data())
    print(type(node_list_dic))

    # Generate edge list
    edges_list = G[num].edges

    # Generate node type list
    node_type_list_num = []
    node_value_type_list = []
    node_value = []

    # print(G[num].nodes[0]['f0'])
    # for i in node_list:
    #     for j in range(total_feature):
    #         feature = 'f' + str(j)
    #         print(feature)
    #         if G[num].nodes[i][feature] == 1:
    #             node_type_list_num.append(j)

    for i in node_list:
        for j in node_feature:
            if 'type' in j and 'value' not in j:
                if G[num].nodes[i][j] == 1:
                    m, n = j.split('e')
                    node_type_list_num.append(n)
            elif 'type' in j and 'value' in j:
                if G[num].nodes[i][j] == 1:
                    m, n = j.split('type')
                    node_value_type_list.append(n)
            elif 'int' in j and 'value' in j:
                if G[num].nodes[i]['value_type0'] == 1:
                    node_value.append(int(G[num].nodes[i][j]))
            elif 'float' in j and 'value' in j:
                if G[num].nodes[i]['value_type1'] == 1:
                    node_value.append(float(G[num].nodes[i][j]))
            elif 'string' in j and 'value' in j:
                if G[num].nodes[i][j] == 1:
                    m, n = j.split('value')
                    node_value.append(n)

    print(node_type_list_num)
    total_type = []
    count = []
    node_type_list = []

    for line in f.readlines():
        line = line.strip('\n')
        type_name, type_ID = line.split(' ')

        total_type.append(type_name)
        count.append(type_ID)

    for i in node_type_list_num:
        p = count.index(str(int(i) + 1))
        new_type = total_type[p]
        node_type_list.append(new_type)

    total_value = []
    count_v = []
    node_value_list = []
    for line in f1.readlines():
        line = line.strip('\n')
        if ': ' in line:
            value_id, value_name = line.split(': ')
            total_value.append(value_name)
            count_v.append(value_id)
    print(node_value)
    for i in node_value:
        if type(i) != int and type(i) != float:
            p = count_v.index(str(int(i) + 1))
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
    print(node_value_list)
    return node_list, edges_list, node_type_list, node_value_list, node_value_type_list

def Visualize(path, kind, dataset, name, num=None,test_batch=None):
    pre_path = '/home/zfk/Documents/graph-generation/debug/Graph/' + path
    path_whole = pre_path + '/graphs_500/'
    if kind == 0:
        prefix = 'GraphRNN_RNN_AST_4_128_pred_%d_1'%(name)
    else:
        prefix = 'GraphRNN_RNN_AST%s_4_128_pred_%d_1'%(kind, name)
    print(prefix)
    format = '.dat'
    file = path_whole+prefix+format
    # file = prefix+str(name)+format
    if num == None:
        for c in range(test_batch):
            node_list, edge_list, node_type_list, node_value_list, node_value_type_list\
                = extract(file, c, dataset)

            dot = graphviz.Graph(comment='Result', format='png')

            # Add node
            flag = 0
            # print(len(node_list))
            # print(len(node_type_list))
            for i in node_list:
                node_id = i
                # print(node_id)
                node = node_type_list[flag]
                node_value = node_value_list[flag]
                
                print(node_id, node)
                node_p = '[{}]'.format((node_id)) + ': ' + node + '\n' + '[{}]'.format(node_value)
                dot.node(str(node_id), node_p)
                flag += 1

            for i in edge_list:
                m, n = i[0], i[1]
                dot.edge(str(m), str(n))

            dot.render("process/%s/%s.gv" % (path, str(name)+'-'+str(c)), view=True)

    else:
        node_list, edge_list, node_type_list, node_value_list, node_value_type_list\
            = extract(file, num, dataset)

        dot = graphviz.Graph(comment='Result',format='png')

        # Add node
        flag = 0
        for i in node_list:
            node_id = i
            # print(node_id)
            node = node_type_list[flag]
            node_value = node_value_list[flag]
            print(node_value)
            print(node_id, node)
            node_p = '[{}]'.format((node_id)) + ': ' + node + '\n' + '[{}]'.format(node_value)
            dot.node(str(node_id), node_p)
            flag += 1

        for i in edge_list:
            m, n = i[0], i[1]
            dot.edge(str(m), str(n))

        dot.render("process/%s/%s.gv" %(path,name), view=True)

if __name__ == '__main__':

    # #TODO: this section depends on choice. (Generate all processes or only the final result)
    #
    for i in range(5000, 8050, 100):
        Visualize('code-0819', '468-50-normalize', '468-50', i,None, 8)
    # Visualize('code-0813-value', '468-50-normalize','468-50', 3000, 0, None)
