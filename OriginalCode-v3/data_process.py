import networkx as nx
import numpy as np


def Graph_load_batch(min_num_nodes = 1, max_num_nodes = 40, name = 'AST'):

    print('Loading graph dataset: ' + str(name))

    G = nx.Graph()
    # path = "dataset/AST/shared/"
    # if name == "200Graphs":
    path = "dataset/AST/zips/200Graphs/"
    path = "../OriginalCode-v4/dataset/"

    data_adj = np.loadtxt(path + name + '_A.txt', delimiter=',').astype(int)
    data_node_label = np.loadtxt(path + name + '_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path + name + '_graph_indicator.txt', delimiter=',').astype(int)
    data_graph_labels = np.loadtxt(path + name + '_graph_labels.txt', delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))
    number_of_nodes = data_node_label.shape[0]
    number_of_node_types = max(data_node_label)

    # print(len(data_tuple))
    # print(data_tuple[0])
    # print(data_node_label)
    # print(number_of_nodes)

    G.add_edges_from(data_tuple)

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

    # print(list(G.nodes(data=True))

    # # Add edges label
    # # Todo: Add edge label：from small number nodes to large number nodes(f1), else(f2)
    #     curr_node_adj = list(G.adj[i+1])
    #     for j in curr_node_adj:
    #         if i < j:
    #             G[i+1][j]['f1'] = 1
    #             G[i+1][j]['f2'] = 0
    #         else:
    #             G[i + 1][j]['f1'] = 0
    #             G[i + 1][j]['f2'] = 1
    #     # This is the version for AST which is undirected graph. For CFG and DFG, i<j f1=1, i>j f2=1
    # # print(list(G.edges(data=True)))

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
    return graphs


if __name__ == "__main__":
    Graph_load_batch()
