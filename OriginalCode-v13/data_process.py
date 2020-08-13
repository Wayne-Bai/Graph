import networkx as nx
import numpy as np
from args import Args

args = Args()

def Graph_load_batch(min_num_nodes = 1, max_num_nodes = 300, name = 'AST'):

    print('Loading graph dataset: ' + str(name))

    G = nx.Graph()
    # path = "dataset/AST/shared/"
    # if name == "200Graphs":
    if args.dataset_type == "2":
        path = "../OriginalCode-v4/dataset_2graphs/"
    elif args.dataset_type == "50":
        path = "../OriginalCode-v8/50-10-30/"
    elif args.dataset_type == "9":
        path = "../dataset/dataset_9graphs_300nodes_30features/"
    elif args.dataset_type == "50-200":
        path = "../dataset/dataset_50graphs_200nodes_25features/"
    elif args.dataset_type == "54":
        path = "../dataset/dataset_54graphs/"
    elif args.dataset_type == "500":
        path = "../dataset/dataset_500graphs_50nodes/"
    elif args.dataset_type == '500-50-normalize':
        path = '../../../dataset/dataset_500graphs_50nodes_normalize/'
    

    data_adj = np.loadtxt(path + name + '_A.txt', delimiter=',').astype(int)
    data_node_label = np.loadtxt(path + name + '_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path + name + '_graph_indicator.txt', delimiter=',').astype(int)
    data_graph_labels = np.loadtxt(path + name + '_graph_labels.txt', delimiter=',').astype(int)
    data_node_value = open(path + name + '_node_value.txt', 'r')

    data_node_label_matrix = list(set(data_node_label))
    # data_node_label_matrix = (np.array(data_node_label_matrix) - 1).tolist()
    data_node_label_mx = [i-1 for i in data_node_label_matrix]
    print(data_node_label_mx)

    print("Loading node type")
    # node_rules_matrix = args.node_rules[[data_node_label_mx]]
    # node_rules_matrix = node_rules_matrix[:, data_node_label_mx]
    # args.node_rules = node_rules_matrix
    # print(args.node_rules)
    # print("node type matrix loaded")


    data_tuple = list(map(tuple, data_adj))
    number_of_nodes = data_node_label.shape[0]
    number_of_node_types = max(data_node_label)
    rule_matrix = args.node_rules[:number_of_node_types+1, :number_of_node_types+1]
    rule_matrix[-1, :] = 0
    rule_matrix[:, -1] = 0
    print(rule_matrix)
    # print(len(data_tuple))
    # print(data_tuple[0])
    # print(data_node_label)
    # print(number_of_nodes)

    G.add_edges_from(data_tuple)
    # print(G.edges())
    # Add node label
    for i in range(number_of_nodes):
        for j in range(number_of_node_types):
            if j == data_node_label[i] -1:
                feature = 'f'+str(j+1)
                G.nodes[i+1][feature] = 1
            else:
                feature = 'f' + str(j+1)
                G.nodes[i+1][feature] = 0
        G.nodes[i+1]['f'+str(number_of_node_types+1)]=0 
        # 0 represent true node, 1 represent there is no node



    # Add edges label
    # Todo: Add edge label：from small number nodes to large number nodes(f1), else(f2)
        curr_node_adj = list(G.adj[i+1])
        # print(curr_node_adj)
        for j in curr_node_adj:
            if i < j:
                G[i+1][j]['f1'] = 0
                G[i+1][j]['f2'] = 1
                G[i+1][j]['f3'] = 0
                G[i+1][j]['f4'] = 0
    # print(G.edges.data())
        # This is the version for AST which is undirected graph. For CFG and DFG, i<j f1=1, i>j f2=1
    # print(list(G.edges(data=True)))

    # add node value
    node_value_list = []
    for line in data_node_value.readlines():
        line = line.strip('\n')
        node_id, node_value = line.split(': ')
        G.nodes[int(node_id)]['value'] = node_value
        n_index, n_value = node_value.split(',')
        if str(n_index) == '3':
            node_value_list.append(int(n_value))
    max_node_value_num = max(node_value_list)
    # print(args.max_node_value_num)

    # print(list(G.nodes(data=True)))

    # Todo: Add graph label: CFG or DFG, vulnerability type
    graph_num = data_graph_indicator.max()
    number_of_graph_types = max(data_graph_labels)
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    # print(node_list)
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # Find the nodes for each graph

        nodes = node_list[data_graph_indicator==i+1]

        G_sub = G.subgraph(nodes)

        # print(G_sub.nodes())

        # print(nodes)
        # print(G_sub.nodes)
        for j in range(number_of_graph_types):
            feature = 'f' + str(j + 1)
            if j == data_graph_labels[i] -1:
                G_sub.graph[feature] = 0    #Todo: It should be modify when we add graph label
            else:
                G_sub.graph[feature] = 0

    # print('nodes', G_sub.number_of_nodes())
    # print('edges', G_sub.number_of_edges())
    # print('label', G_sub.graph)
        if G_sub.number_of_nodes()>=min_num_nodes and G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()

    print('Loaded')


    return graphs, rule_matrix, max_node_value_num


if __name__ == "__main__":
    Graph_load_batch()
