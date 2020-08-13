import torch
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from random import shuffle

import networkx as nx
import pickle as pkl
import scipy.sparse as sp
import logging

import random
import shutil
import os
import time
from model import *
from utils import *

from args import Args
args = Args()

# load ENZYMES and PROTEIN and DD dataset
def Graph_load_batch(min_num_nodes = 20, max_num_nodes = 1000, name = 'ENZYMES',node_attributes = True,graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: '+str(name))
    G = nx.Graph()
    # load data
    path = 'dataset/'+name+'/'
    data_adj = np.loadtxt(path+name+'_A.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path+name+'_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path+name+'_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path+name+'_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path+name+'_graph_labels.txt', delimiter=',').astype(int)


    data_tuple = list(map(tuple, data_adj))
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # print(G.number_of_nodes())
    # print(G.number_of_edges())

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        # print('nodes', G_sub.number_of_nodes())
        # print('edges', G_sub.number_of_edges())
        # print('label', G_sub.graph)
        if G_sub.number_of_nodes()>=min_num_nodes and G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
            # print(G_sub.number_of_nodes(), 'i', i)
    # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
    # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
    print('Loaded')
    return graphs

def test_graph_load_DD():
    graphs, max_num_nodes = Graph_load_batch(min_num_nodes=10,name='DD',node_attributes=False,graph_labels=True)
    shuffle(graphs)
    plt.switch_backend('agg')
    plt.hist([len(graphs[i]) for i in range(len(graphs))], bins=100)
    plt.savefig('figures/test.png')
    plt.close()
    row = 4
    col = 4
    draw_graph_list(graphs[0:row*col], row=row,col=col, fname='figures/test')
    print('max num nodes',max_num_nodes)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# load cora, citeseer and pubmed dataset
def Graph_load(dataset = 'cora'):
    '''
    Load a single graph dataset
    :param dataset: dataset name
    :return:
    '''
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        load = pkl.load(open("dataset/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1')
        # print('loaded')
        objects.append(load)
        # print(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G


######### code test ########
# adj, features,G = Graph_load()
# print(adj)
# print(G.number_of_nodes(), G.number_of_edges())

# _,_,G = Graph_load(dataset='citeseer')
# G = max(nx.strnected_component_subgraphs(G), key=len)
# G = nx.convert_node_labels_to_integers(G)
#
# count = 0
# max_node = 0
# for i in range(G.number_of_nodes()):
#     G_ego = nx.ego_graph(G, i, radius=3)
#     # draw_graph(G_ego,prefix='test'+str(i))
#     m = G_ego.number_of_nodes()
#     if m>max_node:
#         max_node = m
#     if m>=50:
#         print(i, G_ego.number_of_nodes(), G_ego.number_of_edges())
#         count += 1
# print('count', count)
# print('max_node', max_node)




def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    BFS = nx.bfs_tree(G, start_id)
    output = list(BFS)
    # dictionary = dict(nx.bfs_successors(G, start_id))
    # start = [start_id]
    # output = [start_id]
    # while len(start) > 0:
    #     next = []
    #     while len(start) > 0:
    #         current = start.pop(0)
    #         neighbor = dictionary.get(current)
    #         if neighbor is not None:
    #             #### a wrong example, should not permute here!
    #             # shuffle(neighbor)
    #             next = next + neighbor
    #     output = output + next
    #     start = next
    # print(output)
    return output


def add_from_node_f_matrix(matrix, G:nx.Graph):
    N, NF = matrix.shape
    node_idx, f_dict = [], []
    for node in range(N):
        indicator = matrix[node, :]
        if indicator.any() and indicator[-1] == 0:# a node exists
            node_idx.append(node)
            f_dict.append({f'f{feature_idx}':matrix[node,feature_idx] for feature_idx in range(NF)})
    # f_dict = list({f'f{feature_idx}':matrix[node,feature_idx] for feature_idx in range(NF)} for node in range(N))
    node_list = list(zip(node_idx, f_dict))
    G.add_nodes_from(node_list)
    return node_idx


def add_from_edge_f_matrix(matrix, G:nx.Graph, node_idx):
    N, M, EF = matrix.shape
    for i in range(N):
        for j in range(min(M,i+1)):
            if not args.only_use_adj:
                indicator = matrix[i,j,-4:]
                indicator_flag = (indicator.any() and indicator[0] == 0)
            else:
                indicator_flag = matrix[i,j,:].any()
            # if indicator[0] + indicator[1] > 0: # an edge exists
            # if indicator[0] == 1:
            if indicator_flag:# an edge exists
                edge_f_vector = matrix[i,j,:]
                f_dict = {f'f{feature_idx}':edge_f_vector[feature_idx] for feature_idx in range(EF)}
                # if i >=M:
                #     if (i in node_idx) and ((j+i-M+1) in node_idx):
                #         G.add_edges_from([(i, j+i-M+1, f_dict)])
                # else:
                    # To help understand, node numbers: 0(j=4) 1(j=3) 2(j=2) 3(j=1) 4(j=0) 5=i
                i_real_idx = i + 1 # The 0-th row represents node #1
                j_real_idx = i - j #>=0 ==> j<=i
                if (i_real_idx in node_idx) and (j_real_idx in node_idx):
                    G.add_edges_from([(i_real_idx, j_real_idx, f_dict)])


def encode_adj(adj, max_prev_node=10, is_full = False, is_3D=False):
    '''

    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0]-1

    # pick up lower tri
    if is_3D:
        # idx_indicator = np.ones((adj.shape[0], adj.shape[1]))
        # idx_indicator = np.tril(idx_indicator, k=-1)
        # idx_matrix = np.where(idx_indicator==0)
        # adj[idx_matrix] = [1,0,0,0]
        for i in range(adj.shape[0]):
            adj[i, i:] = [1, 0, 0, 0]
    else:
        adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    if is_3D:
        adj = adj[1:n, 0:n - 1, :] # delete first row and last column
    else:
        adj = adj[1:n, 0:n-1] # delete first row and last column

    # use max_prev_node to truncate
    # Now adj is a (n-1)*(n-1) matrix
    if is_3D:
        adj_output = np.zeros((adj.shape[0], max_prev_node, adj.shape[2]))
        adj_output[:, :, 0] = 1
    else:
        adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        if is_3D:
            adj_output[i, output_start:output_end, :] = adj[i, input_start:input_end, :]
            adj_output[i, :, :] = adj_output[i, :, :][::-1, :]  # reverse order
        else:
            adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
            adj_output[i,:] = adj_output[i,:][::-1] # reverse order

    # Append an all-zero row to ensure dimension satisfies
    if is_3D:
        pad = np.zeros((1, max_prev_node, adj.shape[2]))
        pad[:, :, 0] = 1 # [1,0,0,0] as the non-exist edges
        adj_output = np.concatenate((adj_output, pad), axis=0)  # Dim: N * M * EF # 0-th row of dim N represents node#1
    else:
        adj_output = np.concatenate((np.zeros((1,max_prev_node)), adj_output), axis=0) # Dim: N * M # 0-th row of dim N represents node#0
    return adj_output

def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


def encode_adj_flexible(adj):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end-len(adj_slice)+np.amin(non_zero)

    return adj_output



def decode_adj_flexible(adj_output):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    adj = np.zeros((len(adj_output), len(adj_output)))
    for i in range(len(adj_output)):
        output_start = i+1-len(adj_output[i])
        output_end = i+1
        adj[i, output_start:output_end] = adj_output[i]
    adj_full = np.zeros((len(adj_output)+1, len(adj_output)+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full

def test_encode_decode_adj():
######## code test ###########
    G = nx.ladder_graph(5)
    G = nx.grid_2d_graph(20,20)
    G = nx.ladder_graph(200)
    G = nx.karate_club_graph()
    G = nx.connected_caveman_graph(2,3)
    print(G.number_of_nodes())
    
    adj = np.asarray(nx.to_numpy_matrix(G))
    G = nx.from_numpy_matrix(adj)
    #
    start_idx = np.random.randint(adj.shape[0])
    x_idx = np.array(bfs_seq(G, start_idx))
    adj = adj[np.ix_(x_idx, x_idx)]
    
    print('adj\n',adj)
    adj_output = encode_adj(adj,max_prev_node=5)
    print('adj_output\n',adj_output)
    adj_recover = decode_adj(adj_output,max_prev_node=5)
    print('adj_recover\n',adj_recover)
    print('error\n',np.amin(adj_recover-adj),np.amax(adj_recover-adj))
    
    
    adj_output = encode_adj_flexible(adj)
    for i in range(len(adj_output)):
        print(len(adj_output[i]))
    adj_recover = decode_adj_flexible(adj_output)
    print(adj_recover)
    print(np.amin(adj_recover-adj),np.amax(adj_recover-adj))



def encode_adj_full(adj):
    '''
    return a n-1*n-1*2 tensor, the first dimension is an adj matrix, the second show if each entry is valid
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]
    adj_output = np.zeros((adj.shape[0],adj.shape[1],2))
    adj_len = np.zeros(adj.shape[0])

    for i in range(adj.shape[0]):
        non_zero = np.nonzero(adj[i,:])[0]
        input_start = np.amin(non_zero)
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        # write adj
        adj_output[i,0:adj_slice.shape[0],0] = adj_slice[::-1] # put in reverse order
        # write stop token (if token is 0, stop)
        adj_output[i,0:adj_slice.shape[0],1] = 1 # put in reverse order
        # write sequence length
        adj_len[i] = adj_slice.shape[0]

    return adj_output,adj_len

def decode_adj_full(adj_output):
    '''
    return an adj according to adj_output
    :param
    :return:
    '''
    # pick up lower tri
    adj = np.zeros((adj_output.shape[0]+1,adj_output.shape[1]+1))

    for i in range(adj_output.shape[0]):
        non_zero = np.nonzero(adj_output[i,:,1])[0] # get valid sequence
        input_end = np.amax(non_zero)
        adj_slice = adj_output[i, 0:input_end+1, 0] # get adj slice
        # write adj
        output_end = i+1
        output_start = i+1-input_end-1
        adj[i+1,output_start:output_end] = adj_slice[::-1] # put in reverse order
    adj = adj + adj.T
    return adj

def my_decode_adj_cuda(y, node_f, max_n):
    N, M = y.size()
    NF = node_f.size(1)
    vector_list = [Variable(torch.ones(1, NF)).cuda()]
    for i in range(1, N):
        y_i = y.index_select(0, torch.LongTensor([i]).cuda())
        accumulator = Variable(torch.zeros(1, NF)).cuda()
        for j in range(max(0, i-M), max(1, i-1)):
            accumulator += node_f.index_select(0, torch.LongTensor([j]).cuda()).squeeze() * \
            y_i.index_select(1, torch.LongTensor([i-1-j]).cuda())
        vector_list.append(accumulator)
    if N < max_n: # pad 0
        pad = Variable(torch.zeros(max_n - N, NF)).cuda()
        vector_list.append(pad)
    y_prob = torch.cat(vector_list, dim=0)
    return y_prob.view(1, -1, NF)


def generate_test_mask(adj, child, current_node_idx): # output dim: BS * 1 * NF; child_dim: BS * N' * NF
    BS, M = adj.size()
    idx = torch.argmax(adj, dim=1) * (-1) + current_node_idx - 1 
    # just pick up the node with largest idx # assume this node is the parent node.
    mask_list = [child[bs, idx[bs], :].view(1,-1) for bs in range(BS)]
    mask = torch.cat(mask_list, dim=0).view(BS, 1, -1)
    return mask


def test_encode_decode_adj_full():
########### code test #############
    # G = nx.ladder_graph(10)
    G = nx.karate_club_graph()
    # get bfs adj
    adj = np.asarray(nx.to_numpy_matrix(G))
    G = nx.from_numpy_matrix(adj)
    start_idx = np.random.randint(adj.shape[0])
    x_idx = np.array(bfs_seq(G, start_idx))
    adj = adj[np.ix_(x_idx, x_idx)]
    
    adj_output, adj_len = encode_adj_full(adj)
    print('adj\n',adj)
    print('adj_output[0]\n',adj_output[:,:,0])
    print('adj_output[1]\n',adj_output[:,:,1])
    # print('adj_len\n',adj_len)
    
    adj_recover = decode_adj_full(adj_output)
    print('adj_recover\n', adj_recover)
    print('error\n',adj_recover-adj)
    print('error_sum\n',np.amax(adj_recover-adj), np.amin(adj_recover-adj))






########## use pytorch dataloader
class Graph_sequence_sampler_pytorch(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all, self.node_num_all, self.edge_f_all, self.raw_node_f_all, self.len_all = \
            [], [], [], [], []
        self.BFS_first_node = []
        for i,G in enumerate(G_list):
            # add node_type_feature_matrix and edge_type_feature_matrix

            for node in G.nodes():
                if G.nodes[node]['f1'] ==1:
                    first_n = list(G.nodes).index(node)
                    self.BFS_first_node.append(first_n)

            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            node_idx_global = np.asarray(list(G.nodes))
            self.node_num_all.append(node_idx_global)

            # check edge information
            # print("node_num_all: {}".format(self.node_num_all))

            # print(len(G.nodes._nodes), len(G.edges._adjdict), len(list(G.adjacency())))
            self.raw_node_f_all.append(dict(G.nodes._nodes))
            # self.input_node_f_all.append(self.construct_input_node_f(G))
            edge_f_dict = {}
            for k,v in G.edges._adjdict._atlas.items():
                if k in node_idx_global:
                    edge_f_dict[k] = v
            self.edge_f_all.append(edge_f_dict)

            # check edge information
            # print("edge_f_all: {}".format(self.edge_f_all))

            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            print('calculating max previous node, total iteration: {}'.format(iteration))
            self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            print('max previous node: {}'.format(self.max_prev_node))
        else:
            self.max_prev_node = max_prev_node

        # self.max_prev_node = max_prev_node

        # # sort Graph in descending order
        # len_batch_order = np.argsort(np.array(self.len_all))[::-1]
        # self.len_all = [self.len_all[i] for i in len_batch_order]
        # self.adj_all = [self.adj_all[i] for i in len_batch_order]
    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        # Expects 3 outputs:
        #   input_node_f: (N, INF) # INF = M + NF + EF # EF = max_edge_f_num + 2
        #   raw_node_f  : (N,  NF)
        #   edge_f      : (N, M, EF)
        adj_copy = self.adj_all[idx].copy() # Dim: 200 * 200(actual node numbers of this graph: N)

        # check original matrix
        # print("Original Matrix")
        # print(adj_copy)

        node_dict = self.raw_node_f_all[idx].copy()
        node_dict_v = self.raw_node_f_all[idx].copy()
        edge_dict = self.edge_f_all[idx].copy()
        node_num_list = self.node_num_all[idx]
        raw_node_f_batch = self.construct_raw_node_f(node_dict, node_num_list) # Dim: N * NF
        # print('-----------------------------')
        # print(raw_node_f_batch)
        raw_node_v_batch = self.construct_raw_node_v(node_dict_v,node_num_list)# Dim: N * NV (index:3 + int&float:2 + value: 282 = 287)
        # print('*****************************')
        # print(raw_node_v_batch)
        raw_edge_f_batch = self.construct_edge_f(edge_dict, node_num_list) # Dim: N * N * EF
        # print(raw_edge_f_batch)
        edge_f_pooled_batch = self.construct_edge_f(edge_dict, node_num_list, pooling=True)
        # print(edge_f_pooled_batch)
        # print("`````````````````````````````````````````````")

        # x_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        # # x_patch Dim:
        # x_batch[0,:] = 1 # the first input token is all ones
        # y_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        # y_patch Dim:
        # generate input x, y pairs
        len_batch = adj_copy.shape[0] # N
        # x_idx = np.random.permutation(adj_copy.shape[0])
        # adj_copy = adj_copy[np.ix_(x_idx, x_idx)] # re-ordering use x_idx
        adj_copy_matrix = np.asmatrix(adj_copy)
        G = nx.from_numpy_matrix(adj_copy_matrix) # re-generate the graph
        # then do bfs in the permuted G
        # start_idx = np.random.randint(adj_copy.shape[0]) # randomly select a start node


        start_idx = self.BFS_first_node[idx]

        # # check start_idx
        # print("start_idx: {}".format(start_idx))

        x_idx = np.array(bfs_seq(G, start_idx)) # new ordering index vector
        # print("x_idx: {}".format(x_idx))
        # print('*****************************************')

        # # check BFS x_idx
        #
        # print("x_idx: {}".format(x_idx))
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)] # re-ordering use x_idx # Dim of adj_copy: N * N

        # check re-ordering matrix
        # print("re-ordering matrix")
        # print(adj_copy)

        adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node) # Dim: N * 40 (40: max_prev_node, denote as M)
        raw_edge_f_batch = raw_edge_f_batch[np.ix_(x_idx, x_idx)]
        # print("raw_edge_f_batch dim: {}".format(raw_edge_f_batch.shape))
        # print('*****************************************')
        # print(raw_edge_f_batch)
        # print('*****************************************')
        # print('----------------------------------------------')
        edge_f_encoded = encode_adj(raw_edge_f_batch.copy(), max_prev_node=self.max_prev_node, is_3D=True) # Dim: N * M * EF
        # print("edge_f_encoded: {}".format(edge_f_encoded))
        # print('*****************************************')

        # add re-ordering of node_type_feature_matrix and edge_type_feature_matrix
        raw_node_f_batch = raw_node_f_batch[x_idx, :]
        # print("raw_node_f_batch: {}".format(raw_node_f_batch))
        # print('*****************************************')
        # print("edge_f_pooled_batch: {}".format(edge_f_pooled_batch))
        # print('*****************************************')
        edge_f_pooled_batch = edge_f_pooled_batch[x_idx, :]
        # print("edge_f_pooled_batch: {}".format(edge_f_pooled_batch))
        # print('-----------------------------------------')
        if args.not_use_pooling:
            concat_node_f_batch = np.concatenate((adj_encoded, raw_node_f_batch), axis=1)
        else:
            concat_node_f_batch = np.concatenate((adj_encoded, raw_node_f_batch, edge_f_pooled_batch), axis=1)

        # get input_node_f_batch and raw_node_f_batch and edge_f_batch
        # for small graph the rest are zero padded
        x_batch = np.zeros((self.n+1, concat_node_f_batch.shape[1]))  # here zeros are padded for small graph
        x_batch[0, :] = 1  # the first input token is all ones
        # y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:concat_node_f_batch.shape[0] + 1, :] = concat_node_f_batch # has an all-1 row at the beginning of it
        x_batch[concat_node_f_batch.shape[0] + 1:,adj_encoded.shape[1]+raw_node_f_batch.shape[1]-1] = 1

        padded = np.zeros((self.n - raw_node_f_batch.shape[0], raw_node_f_batch.shape[1]))
        padded[:,-1] = 1

        raw_node_f_batch = np.concatenate((raw_node_f_batch,
                                           padded), axis=0)
        smallN, M, EF = edge_f_encoded.shape
        # print(smallN, M, EF)
        # print('-----------------------------------------')
        edge_f_padded_batch = np.zeros((self.n, self.max_prev_node, EF))
        edge_f_padded_batch[:smallN, :M, :] = edge_f_encoded
        if not args.only_use_adj:
            return {'input_node_f':x_batch,'raw_node_f':raw_node_f_batch, 'edge_f':edge_f_padded_batch, 'len':len_batch}
        else:
            adj_encoded_padded_batch = np.zeros((self.n, self.max_prev_node))
            adj_encoded_padded_batch[:(smallN-1), :M] = adj_encoded[1:, :] # 0-th row of adj_encoded represents node #0, exclude
            return {'input_node_f':x_batch,'raw_node_f':raw_node_f_batch, 'edge_f':adj_encoded_padded_batch, 'len':len_batch} #adj_encoded.reshape(-1,adj_encoded.shape[1],1)

    def construct_raw_node_f(self, node_dict, node_num_list):
        node_attr_list = list(next(iter(node_dict.values())).keys())
        N, NF = len(node_dict), len(node_attr_list)-1
        offset = min(node_num_list)
        raw_node_f = np.zeros(shape=(N, NF)) # pad 0 for small graphs
        # idx_list = list(range(N))
        for node, f_dict in node_dict.items():
            f_dict_copy = f_dict.copy()
            del f_dict_copy['value']
            if node in node_num_list:
                raw_node_f[node-offset] = np.asarray(list(f_dict_copy.values())) # 0-indexed

        raw_node_f = raw_node_f[node_num_list-offset,:]
        # raw_node_f[:,-1] = 1
        return raw_node_f

    # for now, we use one-hot to display the string
    def construct_raw_node_v(self, node_dict, node_num_list):
        # node_value_list = list(next(iter(node_dict.values())).values())
        # if 'value' in node_value_list:
        #     print('---------------------------')
        # print(node_value_list)
        # value_list = []
        # for v in node_value_list:
        #     if ',' in str(v):
        #         m,n = v.split(',')
        #         value_list.append(int(n))
        # print(value_list)
        # print(args.max_node_value_num)
        NV = int(args.max_node_value_num)
        N = len(node_dict)
        offset = min(node_num_list)
        index_matrix = np.zeros(shape=(N,3)) # 3 means int, float, string
        raw_node_v_num = np.zeros(shape=(N,2))  # value of int and float
        raw_node_v_str = np.zeros(shape=(N, NV)) # value of string
        # idx_list = list(range(N))
        for node, f_dict in node_dict.items():
            # dic_value = f_dict.pop('value')
            for k,v in f_dict.items():
                if k == 'value':
                    n_index, n_value = v.split(',')
                    if node in node_num_list:
                        if int(n_index) == 1:
                            index_matrix[node-offset, 0] = 1
                            raw_node_v_num[node-offset, 0] = int(n_value)
                        if int(n_index) == 2:
                            index_matrix[node-offset, 1] = 1
                            raw_node_v_num[node-offset, 1] = float(n_value)
                        if int(n_index) == 3:
                            index_matrix[node-offset, 2] = 1
                            raw_node_v_str[node-offset, int(n_value)-1] = 1

        # print('-----------------')
        # print(index_matrix)
        # print(raw_node_v_num)
        print(raw_node_v_str.shape)
        raw_node_v = np.concatenate((index_matrix, raw_node_v_num, raw_node_v_str), axis=1)

        raw_node_v = raw_node_v[node_num_list-offset,:]
        # raw_node_f[:,-1] = 1
        return raw_node_v

    def construct_input_node_f(self, node_dict):
        pass


    def construct_edge_f(self, edge_dict, node_num_list, pooling=False):
        node_edge_dict = next(iter(edge_dict.values()))
        N, EF = len(edge_dict), len(list(next(iter(node_edge_dict.values())).keys()))
        offset = min(node_num_list)
        edge_f = np.zeros(shape=(N, N, EF)) # pad 0 for small graphs
        edge_f[:, :, 0] = 1
        # no_edge = [1, 0, 0, 0]
        l2h_edge = [0, 1, 0, 0]
        h2l_edge = [0, 0, 1, 0]
        duo_edge = [0, 0, 0, 1]
        for node_i, i_edge_dict in edge_dict.items():
            for node_j, edge_f_dict in i_edge_dict.items():
                # if node_i in node_num_list and node_j in node_num_list:
                #     edge_f[node_i-offset][node_j-offset] = np.asarray(list(edge_f_dict.values())) # still 0-indexed!
                if node_i in node_num_list and node_j in node_num_list and node_i < node_j and list(edge_f_dict.values()) == l2h_edge:
                    edge_f[node_i-offset][node_j-offset] = np.asarray(l2h_edge)
                elif node_i in node_num_list and node_j in node_num_list and node_i > node_j and list(edge_f_dict.values()) == l2h_edge:
                    edge_f[node_i-offset][node_j-offset] = np.asarray(h2l_edge)
                elif node_i in node_num_list and node_j in node_num_list and list(edge_f_dict.values()) == duo_edge:
                    edge_f[node_i-offset][node_j-offset] = np.asarray(duo_edge)
                # else:
                #     edge_f[node_i - offset][node_j - offset] = np.asarray(no_edge)


        edge_f = edge_f[np.ix_(node_num_list-offset, node_num_list-offset)] # return dim (N, N, EF)
        # print(edge_f)
        # print("*************************************************")
        if pooling:
            #edge_f = np.sum(edge_f, axis=1) / float(len(node_num_list)) # return dim (N, EF)
            edge_f = np.mean(edge_f, axis=1) # divided by big N

        return edge_f

    def calc_max_prev_node(self, iter=20000,topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print('iter {} times'.format(i))
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-1*topk:]
        return max_prev_node



########## use pytorch dataloader
class Graph_sequence_sampler_pytorch_nobfs(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros((self.n, self.n-1))  # here zeros are padded for small graph
        x_batch[0,:] = 1 # the first input token is all ones
        y_batch = np.zeros((self.n, self.n-1))  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.n-1)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        return {'x':x_batch,'y':y_batch, 'len':len_batch}

# dataset = Graph_sequence_sampler_pytorch_nobfs(graphs)
# print(dataset[1]['x'])
# print(dataset[1]['y'])
# print(dataset[1]['len'])







########## use pytorch dataloader
class Graph_sequence_sampler_pytorch_canonical(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            # print('calculating max previous node, total iteration: {}'.format(iteration))
            # self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            # print('max previous node: {}'.format(self.max_prev_node))
            self.max_prev_node = self.n-1
        else:
            self.max_prev_node = max_prev_node

        # self.max_prev_node = max_prev_node

        # # sort Graph in descending order
        # len_batch_order = np.argsort(np.array(self.len_all))[::-1]
        # self.len_all = [self.len_all[i] for i in len_batch_order]
        # self.adj_all = [self.adj_all[i] for i in len_batch_order]
    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        x_batch[0,:] = 1 # the first input token is all ones
        y_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        # adj_copy_matrix = np.asmatrix(adj_copy)
        # G = nx.from_numpy_matrix(adj_copy_matrix)
        # then do bfs in the permuted G
        # start_idx = G.number_of_nodes()-1
        # x_idx = np.array(bfs_seq(G, start_idx))
        # adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj(adj_copy, max_prev_node=self.max_prev_node)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        return {'x':x_batch,'y':y_batch, 'len':len_batch}

    def calc_max_prev_node(self, iter=20000,topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print('iter {} times'.format(i))
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-1*topk:]
        return max_prev_node



########## use pytorch dataloader
class Graph_sequence_sampler_pytorch_nll(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            adj = np.asarray(nx.to_numpy_matrix(G))
            adj_temp = self.calc_adj(adj)
            self.adj_all.extend(adj_temp)
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            # print('calculating max previous node, total iteration: {}'.format(iteration))
            # self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            # print('max previous node: {}'.format(self.max_prev_node))
            self.max_prev_node = self.n-1
        else:
            self.max_prev_node = max_prev_node

        # self.max_prev_node = max_prev_node

        # # sort Graph in descending order
        # len_batch_order = np.argsort(np.array(self.len_all))[::-1]
        # self.len_all = [self.len_all[i] for i in len_batch_order]
        # self.adj_all = [self.adj_all[i] for i in len_batch_order]
    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        x_batch[0,:] = 1 # the first input token is all ones
        y_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        # adj_copy_matrix = np.asmatrix(adj_copy)
        # G = nx.from_numpy_matrix(adj_copy_matrix)
        # then do bfs in the permuted G
        # start_idx = G.number_of_nodes()-1
        # x_idx = np.array(bfs_seq(G, start_idx))
        # adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj(adj_copy, max_prev_node=self.max_prev_node)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        return {'x':x_batch,'y':y_batch, 'len':len_batch}

    def calc_adj(self,adj):
        max_iter = 10000
        adj_all = [adj]
        adj_all_len = 1
        i_old = 0
        for i in range(max_iter):
            adj_copy = adj.copy()
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            add_flag = True
            for adj_exist in adj_all:
                if np.array_equal(adj_exist, adj_copy):
                    add_flag = False
                    break
            if add_flag:
                adj_all.append(adj_copy)
                adj_all_len += 1
            if adj_all_len % 10 ==0:
                print('adj found:',adj_all_len,'iter used',i)
        return adj_all



# graphs = [nx.barabasi_albert_graph(20,3)]
# graphs = [nx.grid_2d_graph(4,4)]
# dataset = Graph_sequence_sampler_pytorch_nll(graphs)











############## below are codes not used in current version
############## they are based on pytorch default data loader, we should consider reimplement them in current datasets, since they are more efficient


# normal version
class Graph_sequence_sampler_truncate():
    '''
    the output will truncate according to the max_prev_node
    '''
    def __init__(self, G_list, max_node_num=25, batch_size=4, max_prev_node = 25):
        self.batch_size = batch_size
        self.n = max_node_num
        self.max_prev_node = max_prev_node

        self.adj_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))

    def sample(self):
        # batch, length, feature
        x_batch = np.zeros((self.batch_size, self.n, self.max_prev_node)) # here zeros are padded for small graph
        y_batch = np.zeros((self.batch_size, self.n, self.max_prev_node)) # here zeros are padded for small graph
        len_batch = np.zeros(self.batch_size)
        # generate input x, y pairs
        for i in range(self.batch_size):
            # first sample and get a permuted adj
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            len_batch[i] = adj_copy.shape[0]
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)
            # get x and y and adj
            # for small graph the rest are zero padded
            y_batch[i, 0:adj_encoded.shape[0], :] = adj_encoded
            x_batch[i, 1:adj_encoded.shape[0]+1, :] = adj_encoded
        # sort in descending order
        len_batch_order = np.argsort(len_batch)[::-1]
        len_batch = len_batch[len_batch_order]
        x_batch = x_batch[len_batch_order,:,:]
        y_batch = y_batch[len_batch_order,:,:]

        return torch.from_numpy(x_batch).float(), torch.from_numpy(y_batch).float(), len_batch.astype('int').tolist()
    def calc_max_prev_node(self,iter):
        max_prev_node = []
        for i in range(iter):
            if i%(iter/10)==0:
                print(i)
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            time1 = time.time()
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-100:]
        return max_prev_node


# graphs, max_num_nodes = Graph_load_batch(min_num_nodes=6, name='DD',node_attributes=False)
# dataset = Graph_sequence_sampler_truncate([nx.karate_club_graph()])
# max_prev_nodes = dataset.calc_max_prev_node(iter=10000)
# print(max_prev_nodes)
# x,y,len = dataset.sample()
# print('x',x)
# print('y',y)
# print(len)




# only output y_batch (which is needed in batch version of new model)
class Graph_sequence_sampler_fast():
    def __init__(self, G_list, max_node_num=25, batch_size=4, max_prev_node = 25):
        self.batch_size = batch_size
        self.G_list = G_list
        self.n = max_node_num
        self.max_prev_node = max_prev_node

        self.adj_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))


    def sample(self):
        # batch, length, feature
        y_batch = np.zeros((self.batch_size, self.n, self.max_prev_node))  # here zeros are padded for small graph
        # generate input x, y pairs
        for i in range(self.batch_size):
            # first sample and get a permuted adj
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('graph size',adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # get the feature for the permuted G
            # dict = nx.bfs_successors(G, start_idx)
            # print('dict', dict, 'node num', self.G.number_of_nodes())
            # print('x idx', x_idx, 'len', len(x_idx))

            # print('adj')
            # np.set_printoptions(linewidth=200)
            # for print_i in range(adj_copy.shape[0]):
            #     print(adj_copy[print_i].astype(int))
            # adj_before = adj_copy.copy()

            # encode adj
            adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)
            # print('adj encoded')
            # np.set_printoptions(linewidth=200)
            # for print_i in range(adj_copy.shape[0]):
            #     print(adj_copy[print_i].astype(int))


            # decode adj
            # print('adj recover error')
            # adj_decode = decode_adj(adj_encoded.copy(), max_prev_node=self.max_prev_node)
            # adj_err = adj_decode-adj_copy
            # print(np.sum(adj_err))
            # if np.sum(adj_err)!=0:
            #     print(adj_err)
            # np.set_printoptions(linewidth=200)
            # for print_i in range(adj_err.shape[0]):
            #     print(adj_err[print_i].astype(int))

            # get x and y and adj
            # for small graph the rest are zero padded
            y_batch[i, 0:adj_encoded.shape[0], :] = adj_encoded


            # np.set_printoptions(linewidth=200,precision=3)
            # print('y\n')
            # for print_i in range(self.y_batch[i,:,:].shape[0]):
            #     print(self.y_batch[i,:,:][print_i].astype(int))
            # print('x\n')
            # for print_i in range(self.x_batch[i, :, :].shape[0]):
            #     print(self.x_batch[i, :, :][print_i].astype(int))
            # print('adj\n')
            # for print_i in range(self.adj_batch[i, :, :].shape[0]):
            #     print(self.adj_batch[i, :, :][print_i].astype(int))
            # print('adj_norm\n')
            # for print_i in range(self.adj_norm_batch[i, :, :].shape[0]):
            #     print(self.adj_norm_batch[i, :, :][print_i].astype(float))
            # print('feature\n')
            # for print_i in range(self.feature_batch[i, :, :].shape[0]):
            #     print(self.feature_batch[i, :, :][print_i].astype(float))


        # print('x_batch\n',self.x_batch)
        # print('y_batch\n',self.y_batch)

        return torch.from_numpy(y_batch).float()

# graphs, max_num_nodes = Graph_load_batch(min_num_nodes=6, name='PROTEINS_full')
# print(max_num_nodes)
# G = nx.ladder_graph(100)
# # G1 = nx.karate_club_graph()
# # G2 = nx.connected_caveman_graph(4,5)
# G_list = [G]
# dataset = Graph_sequence_sampler_fast(graphs, batch_size=128, max_node_num=max_num_nodes, max_prev_node=30)
# for i in range(5):
#     time0 = time.time()
#     y = dataset.sample()
#     time1 = time.time()
#     print(i,'time', time1 - time0)


# output size is flexible (using list to represent), batch size is 1
class Graph_sequence_sampler_flexible():
    def __init__(self, G_list):
        self.G_list = G_list
        self.adj_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))

        self.y_batch = []
    def sample(self):
        # generate input x, y pairs
        # first sample and get a permuted adj
        adj_idx = np.random.randint(len(self.adj_all))
        adj_copy = self.adj_all[adj_idx].copy()
        # print('graph size',adj_copy.shape[0])
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)
        G = nx.from_numpy_matrix(adj_copy_matrix)
        # then do bfs in the permuted G
        start_idx = np.random.randint(adj_copy.shape[0])
        x_idx = np.array(bfs_seq(G, start_idx))
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        # get the feature for the permuted G
        # dict = nx.bfs_successors(G, start_idx)
        # print('dict', dict, 'node num', self.G.number_of_nodes())
        # print('x idx', x_idx, 'len', len(x_idx))

        # print('adj')
        # np.set_printoptions(linewidth=200)
        # for print_i in range(adj_copy.shape[0]):
        #     print(adj_copy[print_i].astype(int))
        # adj_before = adj_copy.copy()

        # encode adj
        adj_encoded = encode_adj_flexible(adj_copy.copy())
        # print('adj encoded')
        # np.set_printoptions(linewidth=200)
        # for print_i in range(adj_copy.shape[0]):
        #     print(adj_copy[print_i].astype(int))


        # decode adj
        # print('adj recover error')
        # adj_decode = decode_adj(adj_encoded.copy(), max_prev_node=self.max_prev_node)
        # adj_err = adj_decode-adj_copy
        # print(np.sum(adj_err))
        # if np.sum(adj_err)!=0:
        #     print(adj_err)
        # np.set_printoptions(linewidth=200)
        # for print_i in range(adj_err.shape[0]):
        #     print(adj_err[print_i].astype(int))

        # get x and y and adj
        # for small graph the rest are zero padded
        self.y_batch=adj_encoded


        # np.set_printoptions(linewidth=200,precision=3)
        # print('y\n')
        # for print_i in range(self.y_batch[i,:,:].shape[0]):
        #     print(self.y_batch[i,:,:][print_i].astype(int))
        # print('x\n')
        # for print_i in range(self.x_batch[i, :, :].shape[0]):
        #     print(self.x_batch[i, :, :][print_i].astype(int))
        # print('adj\n')
        # for print_i in range(self.adj_batch[i, :, :].shape[0]):
        #     print(self.adj_batch[i, :, :][print_i].astype(int))
        # print('adj_norm\n')
        # for print_i in range(self.adj_norm_batch[i, :, :].shape[0]):
        #     print(self.adj_norm_batch[i, :, :][print_i].astype(float))
        # print('feature\n')
        # for print_i in range(self.feature_batch[i, :, :].shape[0]):
        #     print(self.feature_batch[i, :, :][print_i].astype(float))

        return self.y_batch,adj_copy


# G = nx.ladder_graph(5)
# # G = nx.grid_2d_graph(20,20)
# # G = nx.ladder_graph(200)
# graphs = [G]
#
# graphs, max_num_nodes = Graph_load_batch(min_num_nodes=6, name='ENZYMES')
# sampler = Graph_sequence_sampler_flexible(graphs)
#
# y_max_all = []
# for i in range(10000):
#     y_raw,adj_copy = sampler.sample()
#     y_max = max(len(y_raw[i]) for i in range(len(y_raw)))
#     y_max_all.append(y_max)
#     # print('max bfs node',y_max)
# print('max', max(y_max_all))
# print(y[1])
# print(Variable(torch.FloatTensor(y[1])).cuda(CUDA))











########### potential use: an encoder along with the GraphRNN decoder
# preprocess the adjacency matrix
def preprocess(A):
    # Get size of the adjacency matrix
    size = len(A)
    # Get the degrees for each node
    degrees = np.sum(A, axis=1)+1

    # Create diagonal matrix D from the degrees of the nodes
    D = np.diag(np.power(degrees, -0.5).flatten())
    # Cholesky decomposition of D
    # D = np.linalg.cholesky(D)
    # Inverse of the Cholesky decomposition of D
    # D = np.linalg.inv(D)
    # Create an identity matrix of size x size
    I = np.eye(size)
    # Create A hat
    A_hat = A + I
    # Return A_hat
    A_normal = np.dot(np.dot(D,A_hat),D)
    return A_normal


# truncate the output seqence to save representation, and allowing for infinite generation
# now having a list of graphs
class Graph_sequence_sampler_bfs_permute_truncate_multigraph():
    def __init__(self, G_list, max_node_num=25, batch_size=4, max_prev_node = 25, feature = None):
        self.batch_size = batch_size
        self.G_list = G_list
        self.n = max_node_num
        self.max_prev_node = max_prev_node

        self.adj_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
        self.has_feature = feature

    def sample(self):

        # batch, length, feature
        # self.x_batch = np.ones((self.batch_size, self.n - 1, self.max_prev_node))
        x_batch = np.zeros((self.batch_size, self.n, self.max_prev_node)) # here zeros are padded for small graph
        # self.x_batch[:,0,:] = np.ones((self.batch_size, self.max_prev_node))  # first input is all ones
        # batch, length, feature
        y_batch = np.zeros((self.batch_size, self.n, self.max_prev_node)) # here zeros are padded for small graph
        # batch, length, length
        adj_batch = np.zeros((self.batch_size, self.n, self.n)) # here zeros are padded for small graph
        # batch, size, size
        adj_norm_batch = np.zeros((self.batch_size, self.n, self.n))  # here zeros are padded for small graph
        # batch, size, feature_len: degree and clustering coefficient
        if self.has_feature is None:
            feature_batch = np.zeros((self.batch_size, self.n, self.n)) # use one hot feature
        else:
            feature_batch = np.zeros((self.batch_size, self.n, 2))

        # generate input x, y pairs
        for i in range(self.batch_size):
            time0 = time.time()
            # first sample and get a permuted adj
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            time1 = time.time()
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # get the feature for the permuted G
            node_list = [G.nodes()[i] for i in x_idx]
            feature_degree = np.array(list(G.degree(node_list).values()))[:,np.newaxis]
            feature_clustering = np.array(list(nx.clustering(G,nodes=node_list).values()))[:,np.newaxis]
            time2 = time.time()

            # dict = nx.bfs_successors(G, start_idx)
            # print('dict', dict, 'node num', self.G.number_of_nodes())
            # print('x idx', x_idx, 'len', len(x_idx))

            # print('adj')
            # np.set_printoptions(linewidth=200)
            # for print_i in range(adj_copy.shape[0]):
            #     print(adj_copy[print_i].astype(int))
            # adj_before = adj_copy.copy()

            # encode adj
            adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)
            # print('adj encoded')
            # np.set_printoptions(linewidth=200)
            # for print_i in range(adj_copy.shape[0]):
            #     print(adj_copy[print_i].astype(int))


            # decode adj
            # print('adj recover error')
            # adj_decode = decode_adj(adj_encoded.copy(), max_prev_node=self.max_prev_node)
            # adj_err = adj_decode-adj_copy
            # print(np.sum(adj_err))
            # if np.sum(adj_err)!=0:
            #     print(adj_err)
            # np.set_printoptions(linewidth=200)
            # for print_i in range(adj_err.shape[0]):
            #     print(adj_err[print_i].astype(int))

            # get x and y and adj
            # for small graph the rest are zero padded
            y_batch[i, 0:adj_encoded.shape[0], :] = adj_encoded
            x_batch[i, 1:adj_encoded.shape[0]+1, :] = adj_encoded
            adj_batch[i, 0:adj_copy.shape[0], 0:adj_copy.shape[0]] = adj_copy
            adj_copy_norm = preprocess(adj_copy)
            time3 = time.time()
            adj_norm_batch[i, 0:adj_copy.shape[0], 0:adj_copy.shape[0]] = adj_copy_norm

            if self.has_feature is None:
                feature_batch[i, 0:adj_copy.shape[0], 0:adj_copy.shape[0]] = np.eye(adj_copy.shape[0])
            else:
                feature_batch[i,0:adj_copy.shape[0],:] = np.concatenate((feature_degree,feature_clustering),axis=1)


            # np.set_printoptions(linewidth=200,precision=3)
            # print('y\n')
            # for print_i in range(self.y_batch[i,:,:].shape[0]):
            #     print(self.y_batch[i,:,:][print_i].astype(int))
            # print('x\n')
            # for print_i in range(self.x_batch[i, :, :].shape[0]):
            #     print(self.x_batch[i, :, :][print_i].astype(int))
            # print('adj\n')
            # for print_i in range(self.adj_batch[i, :, :].shape[0]):
            #     print(self.adj_batch[i, :, :][print_i].astype(int))
            # print('adj_norm\n')
            # for print_i in range(self.adj_norm_batch[i, :, :].shape[0]):
            #     print(self.adj_norm_batch[i, :, :][print_i].astype(float))
            # print('feature\n')
            # for print_i in range(self.feature_batch[i, :, :].shape[0]):
            #     print(self.feature_batch[i, :, :][print_i].astype(float))
            time4 = time.time()
            # print('1 ',time1-time0)
            # print('2 ',time2-time1)
            # print('3 ',time3-time2)
            # print('4 ',time4-time3)

        # print('x_batch\n',self.x_batch)
        # print('y_batch\n',self.y_batch)

        return torch.from_numpy(x_batch).float(), torch.from_numpy(y_batch).float(),\
               torch.from_numpy(adj_batch).float(), torch.from_numpy(adj_norm_batch).float(), torch.from_numpy(feature_batch).float()




















# generate own synthetic dataset
def Graph_synthetic(seed):
    G = nx.Graph()
    np.random.seed(seed)
    base = np.repeat(np.eye(5), 20, axis=0)
    rand = np.random.randn(100, 5) * 0.05
    node_features = base + rand

    # # print('node features')
    # for i in range(node_features.shape[0]):
    #     print(np.around(node_features[i], decimals=4))

    node_distance_l1 = np.ones((node_features.shape[0], node_features.shape[0]))
    node_distance_np = np.zeros((node_features.shape[0], node_features.shape[0]))
    for i in range(node_features.shape[0]):
        for j in range(node_features.shape[0]):
            if i != j:
                node_distance_l1[i,j] = np.sum(np.abs(node_features[i] - node_features[j]))
                # print('node distance', node_distance_l1[i,j])
                node_distance_np[i, j] = 1 / np.sum(np.abs(node_features[i] - node_features[j]) ** 2)

    print('node distance max', np.max(node_distance_l1))
    print('node distance min', np.min(node_distance_l1))
    node_distance_np_sum = np.sum(node_distance_np, axis=1, keepdims=True)
    embedding_dist = node_distance_np / node_distance_np_sum

    # generate the graph
    average_degree = 9
    for i in range(node_features.shape[0]):
        for j in range(i + 1, embedding_dist.shape[0]):
            p = np.random.rand()
            if p < embedding_dist[i, j] * average_degree:
                G.add_edge(i, j)

    G.remove_nodes_from(nx.isolates(G))
    print('num of nodes', G.number_of_nodes())
    print('num of edges', G.number_of_edges())

    G_deg = nx.degree_histogram(G)
    G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
    print('average degree', sum(G_deg_sum) / G.number_of_nodes())
    print('average path length', nx.average_shortest_path_length(G))
    print('diameter', nx.diameter(G))
    G_cluster = sorted(list(nx.clustering(G).values()))
    print('average clustering coefficient', sum(G_cluster) / len(G_cluster))
    print('Graph generation complete!')
    # node_features = np.concatenate((node_features, np.zeros((1,node_features.shape[1]))),axis=0)

    return G, node_features

# G = Graph_synthetic(10)



# return adj and features from a single graph
class GraphDataset_adj(torch.utils.data.Dataset):
    """Graph Dataset"""
    def __init__(self, G, features=None):
        self.G = G
        self.n = G.number_of_nodes()
        adj = np.asarray(nx.to_numpy_matrix(self.G))

        # permute adj
        subgraph_idx = np.random.permutation(self.n)
        # subgraph_idx = np.arange(self.n)
        adj = adj[np.ix_(subgraph_idx, subgraph_idx)]

        self.adj = torch.from_numpy(adj+np.eye(len(adj))).float()
        self.adj_norm = torch.from_numpy(preprocess(adj)).float()
        if features is None:
            self.features = torch.Tensor(self.n, self.n)
            self.features = nn.init.eye(self.features)
        else:
            features = features[subgraph_idx,:]
            self.features = torch.from_numpy(features).float()
        print('embedding size', self.features.size())
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        sample = {'adj':self.adj,'adj_norm':self.adj_norm, 'features':self.features}
        return sample

# G = nx.karate_club_graph()
# dataset = GraphDataset_adj(G)
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)
# for data in train_loader:
#     print(data)


# return adj and features from a list of graphs
class GraphDataset_adj_batch(torch.utils.data.Dataset):
    """Graph Dataset"""
    def __init__(self, graphs, has_feature = True, num_nodes = 20):
        self.graphs = graphs
        self.has_feature = has_feature
        self.num_nodes = num_nodes
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        adj_raw = np.asarray(nx.to_numpy_matrix(self.graphs[idx]))
        np.fill_diagonal(adj_raw,0) # in case the self connection already exists

        # sample num_nodes size subgraph
        subgraph_idx = np.random.permutation(adj_raw.shape[0])[0:self.num_nodes]
        adj_raw = adj_raw[np.ix_(subgraph_idx,subgraph_idx)]

        adj = torch.from_numpy(adj_raw+np.eye(len(adj_raw))).float()
        adj_norm = torch.from_numpy(preprocess(adj_raw)).float()
        adj_raw = torch.from_numpy(adj_raw).float()
        if self.has_feature:
            dictionary = nx.get_node_attributes(self.graphs[idx], 'feature')
            features = np.zeros((self.num_nodes, list(dictionary.values())[0].shape[0]))
            for i in range(self.num_nodes):
                features[i, :] = list(dictionary.values())[subgraph_idx[i]]
            # normalize
            features -= np.mean(features, axis=0)
            epsilon = 1e-6
            features /= (np.std(features, axis=0)+epsilon)
            features = torch.from_numpy(features).float()
        else:
            n = self.num_nodes
            features = torch.Tensor(n, n)
            features = nn.init.eye(features)

        sample = {'adj':adj,'adj_norm':adj_norm, 'features':features, 'adj_raw':adj_raw}
        return sample

# return adj and features from a list of graphs, batch size = 1, so that graphs can have various size each time
class GraphDataset_adj_batch_1(torch.utils.data.Dataset):
    """Graph Dataset"""

    def __init__(self, graphs, has_feature=True):
        self.graphs = graphs
        self.has_feature = has_feature

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        adj_raw = np.asarray(nx.to_numpy_matrix(self.graphs[idx]))
        np.fill_diagonal(adj_raw, 0)  # in case the self connection already exists
        n = adj_raw.shape[0]
        # give a permutation
        subgraph_idx = np.random.permutation(n)
        # subgraph_idx = np.arange(n)

        adj_raw = adj_raw[np.ix_(subgraph_idx, subgraph_idx)]

        adj = torch.from_numpy(adj_raw + np.eye(len(adj_raw))).float()
        adj_norm = torch.from_numpy(preprocess(adj_raw)).float()

        if self.has_feature:
            dictionary = nx.get_node_attributes(self.graphs[idx], 'feature')
            features = np.zeros((n, list(dictionary.values())[0].shape[0]))
            for i in range(n):
                features[i, :] = list(dictionary.values())[i]
            features = features[subgraph_idx, :]
            # normalize
            features -= np.mean(features, axis=0)
            epsilon = 1e-6
            features /= (np.std(features, axis=0) + epsilon)
            features = torch.from_numpy(features).float()
        else:
            features = torch.Tensor(n, n)
            features = nn.init.eye(features)

        sample = {'adj': adj, 'adj_norm': adj_norm, 'features': features}
        return sample




# get one node at a time, for a single graph
class GraphDataset(torch.utils.data.Dataset):
    """Graph Dataset"""
    def __init__(self, G, hops = 1, max_degree = 5, vocab_size = 35, embedding_dim = 35, embedding = None,  shuffle_neighbour = True):
        self.G = G
        self.shuffle_neighbour = shuffle_neighbour
        self.hops = hops
        self.max_degree = max_degree
        if embedding is None:
            self.embedding = torch.Tensor(vocab_size, embedding_dim)
            self.embedding = nn.init.eye(self.embedding)
        else:
            self.embedding = torch.from_numpy(embedding).float()
        print('embedding size', self.embedding.size())
    def __len__(self):
        return len(self.G.nodes())
    def __getitem__(self, idx):
        idx = idx+1
        idx_list = [idx]
        node_list = [self.embedding[idx].view(-1, self.embedding.size(1))]
        node_count_list = []
        for i in range(self.hops):
            # sample this hop
            adj_list = np.array([])
            adj_count_list = np.array([])
            for idx in idx_list:
                if self.shuffle_neighbour:
                    adj_list_new = list(self.G.adj[idx - 1])
                    random.shuffle(adj_list_new)
                    adj_list_new = np.array(adj_list_new) + 1
                else:
                    adj_list_new = np.array(list(self.G.adj[idx-1]))+1
                adj_count_list_new = np.array([len(adj_list_new)])
                adj_list = np.concatenate((adj_list, adj_list_new), axis=0)
                adj_count_list = np.concatenate((adj_count_list, adj_count_list_new), axis=0)
            # print(i, adj_list)
            # print(i, embedding(Variable(torch.from_numpy(adj_list)).long()))
            index = torch.from_numpy(adj_list).long()
            adj_list_emb = self.embedding[index]
            node_list.append(adj_list_emb)
            node_count_list.append(adj_count_list)
            idx_list = adj_list


        # padding, used as target
        idx_list = [idx]
        node_list_pad = [self.embedding[idx].view(-1, self.embedding.size(1))]
        node_count_list_pad = []
        node_adj_list = []
        for i in range(self.hops):
            adj_list = np.zeros(self.max_degree ** (i + 1))
            adj_count_list = np.ones(self.max_degree ** (i)) * self.max_degree
            for j, idx in enumerate(idx_list):
                if idx == 0:
                    adj_list_new = np.zeros(self.max_degree)
                else:
                    if self.shuffle_neighbour:
                        adj_list_new = list(self.G.adj[idx - 1])
                        # random.shuffle(adj_list_new)
                        adj_list_new = np.array(adj_list_new) + 1
                    else:
                        adj_list_new = np.array(list(self.G.adj[idx-1]))+1
                start_idx = j * self.max_degree
                incre_idx = min(self.max_degree, adj_list_new.shape[0])
                adj_list[start_idx:start_idx + incre_idx] = adj_list_new[:incre_idx]
            index = torch.from_numpy(adj_list).long()
            adj_list_emb = self.embedding[index]
            node_list_pad.append(adj_list_emb)
            node_count_list_pad.append(adj_count_list)
            idx_list = adj_list
            # calc adj matrix
            node_adj = torch.zeros(index.size(0),index.size(0))
            for first in range(index.size(0)):
                for second in range(first, index.size(0)):
                    if index[first]==index[second]:
                        node_adj[first,second] = 1
                        node_adj[second,first] = 1
                    elif self.G.has_edge(index[first],index[second]):
                        node_adj[first, second] = 0.5
                        node_adj[second, first] = 0.5
            node_adj_list.append(node_adj)


        node_list = list(reversed(node_list))
        node_count_list = list(reversed(node_count_list))
        node_list_pad = list(reversed(node_list_pad))
        node_count_list_pad = list(reversed(node_count_list_pad))
        node_adj_list = list(reversed(node_adj_list))
        sample = {'node_list':node_list, 'node_count_list':node_count_list,
                  'node_list_pad':node_list_pad, 'node_count_list_pad':node_count_list_pad, 'node_adj_list':node_adj_list}
        return sample


