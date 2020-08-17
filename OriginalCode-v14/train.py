import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm

from utils import *
from model import *
from data import *
from args import Args
import create_graphs

args = Args()

# show all value in the matrix
torch.set_printoptions(profile='full', threshold=np.inf)
np.set_printoptions(threshold=np.inf)

def train_vae_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).cuda()
        y = Variable(y).cuda()

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        y_pred,z_mu,z_lsgms = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        z_mu = pack_padded_sequence(z_mu, y_len, batch_first=True)
        z_mu = pad_packed_sequence(z_mu, batch_first=True)[0]
        z_lsgms = pack_padded_sequence(z_lsgms, y_len, batch_first=True)
        z_lsgms = pad_packed_sequence(z_lsgms, batch_first=True)[0]
        # use cross entropy loss
        loss_bce = binary_cross_entropy_weight(y_pred, y)
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= y.size(0)*y.size(1)*sum(y_len) # normalize
        loss = loss_bce + loss_kl
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        z_mu_mean = torch.mean(z_mu.data)
        z_sgm_mean = torch.mean(z_lsgms.mul(0.5).exp_().data)
        z_mu_min = torch.min(z_mu.data)
        z_sgm_min = torch.min(z_lsgms.mul(0.5).exp_().data)
        z_mu_max = torch.max(z_mu.data)
        z_sgm_max = torch.max(z_lsgms.mul(0.5).exp_().data)


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train bce loss: {:.6f}, train kl loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss_bce.data[0], loss_kl.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))
            print('z_mu_mean', z_mu_mean, 'z_mu_min', z_mu_min, 'z_mu_max', z_mu_max, 'z_sgm_mean', z_sgm_mean, 'z_sgm_min', z_sgm_min, 'z_sgm_max', z_sgm_max)

        # logging
        log_value('bce_loss_'+args.fname, loss_bce.data[0], epoch*args.batch_ratio+batch_idx)
        log_value('kl_loss_' +args.fname, loss_kl.data[0], epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_mean_'+args.fname, z_mu_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_min_'+args.fname, z_mu_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_max_'+args.fname, z_mu_max, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_mean_'+args.fname, z_sgm_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_min_'+args.fname, z_sgm_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_max_'+args.fname, z_sgm_max, epoch*args.batch_ratio + batch_idx)

        loss_sum += loss.data[0]
    return loss_sum/(batch_idx+1)

def test_vae_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False, sample_time = 1):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step, _, _ = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)


    return G_pred_list


def test_vae_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step, _, _ = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].cuda(), current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list



def train_mlp_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).cuda()
        y = Variable(y).cuda()

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)

        loss_sum += loss.data[0]
    return loss_sum/(batch_idx+1)


def test_mlp_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False,sample_time=1):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)


    # # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)
    return G_pred_list



def test_mlp_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].cuda(), current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def test_mlp_partial_simple_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised_simple(y_pred_step, y[:,i:i+1,:].cuda(), current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def train_mlp_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).cuda()
        y = Variable(y).cuda()

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss

        loss = 0
        for j in range(y.size(1)):
            # print('y_pred',y_pred[0,j,:],'y',y[0,j,:])
            end_idx = min(j+1,y.size(2))
            loss += binary_cross_entropy_weight(y_pred[:,j,0:end_idx], y[:,j,0:end_idx])*end_idx


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)

        loss_sum += loss.data[0]
    return loss_sum/(batch_idx+1)





## too complicated, deprecated
# def test_mlp_partial_bfs_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
#     rnn.eval()
#     output.eval()
#     G_pred_list = []
#     for batch_idx, data in enumerate(data_loader):
#         x = data['x'].float()
#         y = data['y'].float()
#         y_len = data['len']
#         test_batch_size = x.size(0)
#         rnn.hidden = rnn.init_hidden(test_batch_size)
#         # generate graphs
#         max_num_node = int(args.max_num_node)
#         y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
#         y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
#         x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
#         for i in range(max_num_node):
#             # 1 back up hidden state
#             hidden_prev = Variable(rnn.hidden.data).cuda()
#             h = rnn(x_step)
#             y_pred_step = output(h)
#             y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
#             x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].cuda(), current=i, y_len=y_len, sample_time=sample_time)
#             y_pred_long[:, i:i + 1, :] = x_step
#
#             rnn.hidden = Variable(rnn.hidden.data).cuda()
#
#             print('finish node', i)
#         y_pred_data = y_pred.data
#         y_pred_long_data = y_pred_long.data.long()
#
#         # save graphs as pickle
#         for i in range(test_batch_size):
#             adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
#             G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
#             G_pred_list.append(G_pred)
#     return G_pred_list


def train_rnn_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output,
                    node_f_gen=None, edge_f_gen=None):
    flag_gen = False
    if node_f_gen : #and edge_f_gen:
        flag_gen = True
    rnn.train()
    output.train()
    if flag_gen:
        node_f_gen.train()
        #edge_f_gen.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader): # Fetch graphs of one batch_size; e.g. 32 graphs
        rnn.zero_grad()
        output.zero_grad()
        if flag_gen:
            node_f_gen.zero_grad()
            #edge_f_gen.zero_grad()
        # x_unsorted = data['x'].float() # Dim: BS * N_max * M (N_max: max node numbers in all graphs)
        # y_unsorted = data['y'].float()
        # add feature matrix, e.g. data['x_node_f']
        # 'input_node_f':x_batch,'raw_node_f':raw_node_f_batch, 'edge_f':edge_f_padded_batch, 'len':len_batch
        input_node_f_unsorted = data['input_node_f'].float() # Dim: BS * N_max * INF
        raw_node_f_unsorted = data['raw_node_f'].float() # Dim: BS * N_max * NF
        edge_f_unsorted = data['edge_f'].float() # Dim: BS * N_max * M * EF
        y_len_unsorted = data['len'] # list of node numbers in each graph in this batch
        y_len_max = max(y_len_unsorted) # denote as N
        # x_unsorted = x_unsorted[:, 0:y_len_max, :]# Dim: BS * N * M
        # y_unsorted = y_unsorted[:, 0:y_len_max, :]# Dim: BS * N * M
        input_node_f_unsorted = input_node_f_unsorted[:, 0:y_len_max, :] # Dim: BS * (N+1) * INF
        raw_node_f_unsorted = raw_node_f_unsorted[:, 0:y_len_max, :] # Dim: BS * N * NF
        
        if not args.only_use_adj:
            BS, N, M, EF = edge_f_unsorted.shape
            edge_f_unsorted = edge_f_unsorted[:, 0:y_len_max, :, :] # Dim: BS * N * M * EF
        else:
            BS, N, M = edge_f_unsorted.shape; EF=1
            edge_f_unsorted = edge_f_unsorted[:, 0:y_len_max, :] # Dim: BS * N * M
        # initialize GRU hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=input_node_f_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input # The graph with most node numbers come first
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        # x = torch.index_select(x_unsorted,0,sort_index) # Dim: BS * N * M
        # y = torch.index_select(y_unsorted,0,sort_index) # Dim: BS * N * M
        input_node_f = torch.index_select(input_node_f_unsorted, 0, sort_index)
        raw_node_f = torch.index_select(raw_node_f_unsorted, 0, sort_index)
        edge_f = torch.index_select(edge_f_unsorted, 0, sort_index)


        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        # y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data # Dim: SumN * M
        # input should be edge_f, output should be dim: SumN * M * EF
        edge_f_reshape = pack_padded_sequence(edge_f,y_len,batch_first=True).data # SumN * M * EF

        # # reverse y_reshape, so that their lengths are sorted, add dimension
        # idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        # idx = torch.LongTensor(idx)
        # y_reshape = y_reshape.index_select(0, idx)
        # y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1) # Dim: SumN * M * 1
        #
        # output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1) # should have all-1 row
        # reverse edge_f_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(edge_f_reshape.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        edge_f_reshape = edge_f_reshape.index_select(0, idx)
        edge_f_reshape = edge_f_reshape.view(edge_f_reshape.size(0), edge_f_reshape.size(1), args.edge_feature_output_dim)  # Dim: SumN * M * (EF or 1)

        edge_rnn_input = torch.cat((torch.ones(edge_f_reshape.size(0), 1, edge_f_reshape.size(2)), edge_f_reshape[:, 0:-1, :]),
                             dim=1)  # should have all-1 row
        # Dim: SumN * (M+1) * EF

        # output_y = y_reshape # Dim: SumN * M * 1
        output_y = edge_f_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,M)]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
            # TODO: understand what's going on

        # pack into variable
        # x = Variable(x).cuda() # Dim should be BS * N * (M + NF + EF)
        # y = Variable(y).cuda()
        # output_x = Variable(output_x).cuda() # Dim should be SumN * M * EF
        output_y = Variable(output_y).cuda() # Dim should be SumN * M * EF

        edge_rnn_input = Variable(edge_rnn_input).cuda()
        input_node_f = Variable(input_node_f).cuda()

        # output_node_f = Variable(torch.zeros(x.size(0), x.size(1), args.max_node_feature_num)).cuda() # Dim should be BS * N * NF
        # if args.loss_type == "mse":
        #     output_node_f = Variable(raw_node_f).cuda()
        # else:
        #     output_node_f = Variable(np.argmax(raw_node_f,axis=-1)).cuda()
        output_node_f = Variable(raw_node_f).cuda()
        # output_edge_f = Variable(torch.zeros(output_y.size(0), output_y.size(1), args.edge_feature_output_dim)).cuda()


        # if using ground truth to train
        # h = rnn(x, pack=True, input_len=y_len) # Dim should be BS * N * hidden_size_rnn_output
        h = rnn(input_node_f, pack=True, input_len=y_len) # Dim: BS * (N+1) * hidden_size_rnn_output

        node_f_pred = node_f_gen(h)  # Dim: BS * (N+1) * NF
        # TODO node_f_pred = Mask & node_f_pred
        # Matrix, (NF, NF) 1,1,1,0,0,0...
        # node_f_pred = torch.softmax(node_f_pred, dim=2) # Dim: BS * N * NF


        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # Dim should be SumN * hidden_size_rnn_output

        # reverse h # TODO: why reverse?
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).cuda()
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, SumN, hidden_size
        # y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred_origin = output(edge_rnn_input, pack=True, input_len=output_y_len) # Dim: SumN * (M+1) * EF
        # edge_f_pred = edge_f_gen(y_pred)  # TODO: check if dim correct
        # edge_f_pred = torch.sigmoid(edge_f_pred)

        # y_pred = torch.softmax(y_pred, dim=2) # Dim: SumN * M * EF

        # clean
        # If all elements in output_y_len are equal to M, this code segment has no effect
        # print(y_pred)
        # print(type(y_pred))
        # print(y_pred.shape)
        y_pred = pack_padded_sequence(y_pred_origin, output_y_len, batch_first=True)
        # print(y_pred)
        # print(type(y_pred))
        # print(y_pred.data.shape)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]


        if args.if_add_train_mask:
            # Add mask according to adj
            # pick node numbers of each graph according to values of each element in y_len
            child_node_f_info = torch.matmul(node_f_pred, torch.FloatTensor(args.node_rules).cuda())
            # Unpack y_pred according to y_len. 
            accumulator = 0
            mask_list = []
            y_pred_untrain = torch.tensor(y_pred.data, requires_grad=False).cuda()
            for idx, each in enumerate(y_len): 
                y_pred_select = y_pred_untrain.index_select(dim=0, index=torch.LongTensor(list(range(accumulator, accumulator + each))).cuda())
                y_pred_select = y_pred_select.index_select(dim=2, index=torch.LongTensor([2]).cuda())
                # [2] means receiving edge # TODO: perhaps should add [3] which is bi-directional
                adj_prob_from_y_pred = torch.sum(y_pred_select, dim=2)

                child_info_batch = child_node_f_info.index_select(dim=0, index=torch.LongTensor([idx]).cuda()).squeeze()
                node_f_pred_batch = my_decode_adj_cuda(adj_prob_from_y_pred, child_info_batch, node_f_pred.size(1))
                accumulator += each
                #if idx != 0:
                mask_list.append(node_f_pred_batch)
            
            mask_new = torch.cat(mask_list, dim=0)
            node_f_pred_new = torch.mul(mask_new, node_f_pred)
        else:
            node_f_pred_new = node_f_pred



        # use cross entropy loss
       
        loss = 0
        log_all_loss = False
        if args.loss_type == "mse":
            direction_loss = my_cross_entropy(y_pred,output_y)
            edge_f_loss = 0
            node_f_loss = my_cross_entropy(node_f_pred, output_node_f)
        else:
        #  direction_loss =
            # print(node_f_pred.shape)
            # print(output_node_f.shape)
            # print(output_y.shape)
            # direction_loss = binary_cross_entropy_weight(F.sigmoid(y_pred[:,:,-2:]),output_y[:,:,-2:])
            # direction_loss = binary_cross_entropy_weight(torch.sigmoid(y_pred[:,:,-2:-1]),output_y[:,:,-2:-1]) 
            # compute loss of last two dimension separately
            # direction_loss = my_cross_entropy(torch.sigmoid(y_pred[:,:,-4:]),output_y[:,:,-4:],if_CE=True)
            if not args.only_use_adj:
                # direction_loss = my_cross_entropy(y_pred[:,:,-4:], torch.argmax(output_y[:,:,-4:],dim=2),if_CE=True)
                direction_loss = my_cross_entropy(y_pred[:,:,-4:], torch.argmax(output_y[:,:,-4:],dim=2),if_CE=True, mask_len=output_y_len)
            else:
                direction_loss = binary_cross_entropy_weight(y_pred, output_y)
            # weights = torch.FloatTensor([0.0, 0, 1.0, 0]).cuda()
            # direction_loss = my_cross_entropy(y_pred[:,:,-4:], torch.argmax(output_y[:,:,-4:],dim=2),if_CE=True,my_weight=weights)
            # y_pred_no_padding = pack_padded_sequence(y_pred[:,:,-4:], output_y_len, batch_first=True)
            # y_groundtruth_no_padding = pack_padded_sequence(torch.argmax(output_y[:,:,-4:],dim=2), output_y_len, batch_first=True)
            # direction_loss = my_cross_entropy(y_pred_no_padding, y_groundtruth_no_padding,if_CE=True)

            # edge_f_loss = my_cross_entropy(y_pred[:,:,:-2], torch.argmax(output_y[:,:,:-2],dim=2))
            edge_f_loss = 0

            # node_f_loss contains: type_loss, index_loss, int_and_float_v_loss, string_loss
            type_loss_end = args.max_node_type_num
            type_loss = my_cross_entropy(node_f_pred_new[:,:,:type_loss_end], \
                torch.argmax(output_node_f[:,:,:type_loss_end],dim=2), if_CE=True, mask_len=y_len)

            index_loss_start = args.max_node_type_num
            index_loss_end = args.max_node_type_num + args.node_index_num
            index_loss = my_cross_entropy(node_f_pred_new[:,:,index_loss_start:index_loss_end], \
                torch.argmax(output_node_f[:,:,index_loss_start:index_loss_end],dim=2), if_CE=True, mask_len=y_len)
            
            ifv_loss_start = args.max_node_type_num + args.node_index_num
            ifv_loss_end = args.max_node_type_num + args.node_index_num + args.node_int_and_float_num
            int_and_float_v_loss = my_cross_entropy(node_f_pred_new[:,:,ifv_loss_start:ifv_loss_end], \
                output_node_f[:,:,ifv_loss_start:ifv_loss_end], if_CE=False, mask_len=y_len)
            
            string_loss = my_cross_entropy(node_f_pred_new[:,:,ifv_loss_end:], \
                torch.argmax(output_node_f[:,:,ifv_loss_end:],dim=2), if_CE=True, mask_len=y_len)

            log_all_loss = True
            node_f_loss_list = [type_loss, index_loss, int_and_float_v_loss[0], int_and_float_v_loss[1], string_loss]
            node_f_loss = 0
            for loss_index, loss_w in enumerate(args.node_loss_w_list):
                node_f_loss += loss_w * node_f_loss_list[loss_index]
            # node_f_loss = my_cross_entropy(node_f_pred_new, output_node_f,if_CE=True) #+ \
               # binary_cross_entropy_weight(edge_f_pred, output_edge_f)
        loss = args.edge_loss_w * (edge_f_loss + direction_loss) + args.node_loss_w * node_f_loss 
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            if not log_all_loss:
                print('Epoch: {}/{}, train loss: {:.6f}, node_f_loss: {:.6f}, edge_f_loss: {:.6f}, direction_loss:{:.6f}, num_layer: {}, hidden: {}'.format(
                    # epoch, args.epochs,loss.data, node_f_loss.data, edge_f_loss.data, args.num_layers, args.hidden_size_rnn))
                    epoch, args.epochs,loss.data, node_f_loss.data, edge_f_loss.data, direction_loss.data, args.num_layers, args.hidden_size_rnn))
            else:
                print('Epoch: {}/{}, train loss: {:.6f}, node_f_loss: {:.6f}, [type_loss {:.6f}, index_loss {:.6f}, \
                    int_and_float_v_loss[0] {:.6f}, int_and_float_v_loss[1] {:.6f}, string_loss {:.6f}], direction_loss:{:.6f}'.format(
                    epoch, args.epochs,loss.data, node_f_loss.data, type_loss.data, index_loss.data, int_and_float_v_loss[0].data, \
                        int_and_float_v_loss[1].data, string_loss.data, direction_loss.data))

        # logging
        log_value('loss_'+args.fname, loss.data, epoch*args.batch_ratio+batch_idx)
        feature_dim = N*M
        loss_sum += loss.data*feature_dim
    return loss_sum/(batch_idx+1)



def test_rnn_epoch(epoch, args, rnn, output, node_f_gen=None, edge_f_gen=None, test_batch_size=16, test_set=None):
    flag_node_f_gen = False
    if node_f_gen:
        flag_node_f_gen = True
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()
    if flag_node_f_gen:
        node_f_gen.eval()

    # # generate graphs
    # max_num_node = int(args.max_num_node)
    # # y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
    # # (32, 361, 40). Should be: node_f_pred_long: (BS, Nmax, NF), edge_f_pred_long: (BS, )
    # node_f_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_node_feature_num)).cuda()
    # #node_f_pred_long[:, :, :] = 1
    # edge_f_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node, args.edge_feature_output_dim)).cuda()
    # x_step = Variable(torch.ones(test_batch_size,1,args.node_feature_input_dim)).cuda()
    # Rnn should start with all-one input
    # (32, 1, 40)
    if args.if_test_use_groundtruth:
        assert not test_set is None
        for _, test_data in enumerate(test_set):
            #TODO: if total_size != batch_size maybe need tab
            # Initialize
            max_num_node = int(args.max_num_node)
            node_f_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_node_feature_num)).cuda()
            #node_f_pred_long[:, :, :] = 1
            edge_f_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node, args.edge_feature_output_dim)).cuda()
            x_step = Variable(torch.ones(test_batch_size,1,args.node_feature_input_dim)).cuda()

            # Pre-processing of test_data
            input_node_f_unsorted = test_data['input_node_f'].float() # Dim: BS * N_max * INF
            
            edge_f_unsorted = test_data['edge_f'].float() # Dim: BS * N_max * M * EF
            y_len_unsorted = test_data['len'] # list of node numbers in each graph in this batch
            y_len_max = max(y_len_unsorted) # denote as N
            
            input_node_f_unsorted = input_node_f_unsorted[:, 1:y_len_max+1, :] # Dim: BS * (N+1) * INF
            # index start from 1, ignore the all-1 row
            
            if not args.only_use_adj:
                BS, N, M, EF = edge_f_unsorted.shape
                edge_f_unsorted = edge_f_unsorted[:, 0:y_len_max, :, :] # Dim: BS * N * M * EF
            else:
                BS, N, M = edge_f_unsorted.shape; EF=1
                edge_f_unsorted = edge_f_unsorted[:, 0:y_len_max, :] # Dim: BS * N * M
            # initialize GRU hidden state according to batch size
            rnn.hidden = rnn.init_hidden(batch_size=input_node_f_unsorted.size(0))
            
            # sort input # The graph with most node numbers come first
            y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
            y_len = y_len.numpy().tolist()
            input_node_f = torch.index_select(input_node_f_unsorted, 0, sort_index)
            edge_f = torch.index_select(edge_f_unsorted, 0, sort_index)

            for i in range(max_num_node):
                h = rnn(x_step) # Dim: (BS, 1, Hidden)
                # if i == -1:
                #     x_step = Variable(torch.rand(test_batch_size,1,args.node_feature_input_dim)).cuda()
                #     continue
                node_f_pred_step = node_f_gen(h)
                # node_f_pred_step = torch.softmax(node_f_pred_step, dim=2) # Dim: (BS, 1, Hidden)
                # node_f_pred_step = sample_sigmoid(node_f_pred_step, sample=False, thresh=args.test_thres, sample_time=1)
                # TODO node_f_pred = Mask & node_f_pred

                # Add h reverse for testing
                # idx = [i for i in range(h.size(0) - 1, -1, -1)]
                # idx = Variable(torch.LongTensor(idx)).cuda()
                # h = h.index_select(0, idx)

                hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).cuda()
                output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                        dim=0)  # num_layers, batch_size, hidden_size
                # should renew x_step
                x_step = Variable(torch.zeros(test_batch_size,1,args.node_feature_input_dim)).cuda()
                # x_step[:, :, args.max_prev_node:args.max_node_feature_num+args.max_prev_node] = node_f_pred_step

                output_x_step = Variable(torch.ones(test_batch_size,1,args.edge_feature_output_dim)).cuda()
                # Rnn should start with all-one input
                for j in range(min(args.max_prev_node,i+1)):
                    output_y_pred_step = output(output_x_step) # (BS, 1, EF)
                    # if j == -1:
                    #     output_x_step = Variable(torch.rand(test_batch_size,1,args.edge_feature_output_dim)).cuda()
                    #     continue
                    # output_y_pred_step = torch.softmax(output_y_pred_step, dim=2)
                    output_x_step = sample_sigmoid(output_y_pred_step, sample=False, thresh=args.test_thres, sample_time=1,if_sigmoid=args.only_use_adj)
                    # x_step[:,:,j:j+1] = output_x_step
                    edge_f_pred_long[:, i:i + 1, j:j+1, :] = output_x_step.view(output_x_step.size(0), output_x_step.size(1),
                                                                                1, output_x_step.size(2))
                    output.hidden = Variable(output.hidden.data).cuda()

                    # Change some values of output_x_step to groundtruth
                    # for bs in range(args.test_batch_size):
                    #     if (i+1) < y_len[bs]: # Should use groundtruth
                    #         output_x_step[bs,0,:] = edge_f[bs,i,j,:]
                            # if bs==0:
                            #     print(f'output_x_step (i+1)-j: {i+1,j} value: {output_x_step[bs,0,2]}')
                            
                    # Add noise to output_x_step
                    #output_x_step += Variable(torch.rand(output_x_step.size(0),output_x_step.size(1),output_x_step.size(2))).cuda()
                # y_pred_long[:, i:i + 1, :] = x_step

                
                if args.if_add_test_mask and i != 0:
                    child_node_f = torch.matmul(node_f_pred_long[:, :i, :], torch.FloatTensor(args.node_rules).cuda()) # Dim: BS * N' * NF
                    indicator_idx = 2 if not args.only_use_adj else 0
                    node_f_test_mask = generate_test_mask(edge_f_pred_long[:, i-1, :, indicator_idx], child_node_f, i)
                    node_f_pred_step = node_f_test_mask * node_f_pred_step # Dim: BS * 1 * NF
                #elif not args.if_add_test_mask
                if node_f_pred_step.bool().any():
                    node_f_pred_step = sample_sigmoid(node_f_pred_step, sample=False, thresh=args.test_thres, sample_time=1)
                else:
                    node_f_pred_step = Variable(torch.zeros(node_f_pred_step.size(0),node_f_pred_step.size(1),node_f_pred_step.size(2))).cuda()
                    node_f_pred_step[:,:,-1] = 1
                
                rnn.hidden = Variable(rnn.hidden.data).cuda()
                
                x_step[:, :, args.max_prev_node:args.max_node_feature_num+args.max_prev_node] = node_f_pred_step
                node_f_pred_long[:, i:i+1, :] = node_f_pred_step
                node_edge_info = edge_f_pred_long[:, i-1, :, :] # (BS,  M, EF) # where EF = args.edge_feature_output_dim = args.max_edge_feature_num + 2
                # Here i-1 is correct!!! e.g.: node num i=1 should look for 0-th row in edge_f_pred_long
                
                indicator_idx = -3 if not args.only_use_adj else -1
                x_step[:, :, :args.max_prev_node] = \
                    torch.tensor(torch.tensor(node_edge_info[:, :, indicator_idx:], dtype=torch.bool).any(2), dtype=torch.uint8).\
                        view(test_batch_size, 1, args.max_prev_node) # (BS, 1, M)
                        # An edge exists only when last three values are 0
                if not args.not_use_pooling:
                    x_step[:, :, args.max_node_feature_num+args.max_prev_node:] = \
                    node_edge_info.mean(dim=1, keepdim=True) # (BS, 1, EF)
                
                # Change some vaules of x_step to groundtruth
                for bs in range(args.test_batch_size):
                    if i < y_len[bs]: # Should use groundtruth
                        # if random.randint(0,1) == 1:
                        x_step[bs,0,:] = input_node_f[bs,i,:]
                        # if bs==0:
                        #     print(f'x_step i: {i} value: {torch.argmax(x_step[bs,0,args.max_prev_node:args.max_node_feature_num+args.max_prev_node],dim=-1)+1}')
                        
                # Add random noise to x_step
                #x_step += Variable(torch.rand(x_step.size(0),x_step.size(1),x_step.size(2))).cuda()
                
            # y_pred_long_data = y_pred_long.data.long()
    
        #TODO: if total_size != batch_size maybe need tab
        node_f_pred_long_data = node_f_pred_long.data.float()
        edge_f_pred_long_data = edge_f_pred_long.data.float()

        #TODO: if total_size != batch_size maybe need tab
        # save graphs as pickle
        G_pred_list = []
        for i in range(test_batch_size):
            # adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            # G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred = nx.Graph()
            node_idx_list = add_from_node_f_matrix(node_f_pred_long_data[i].cpu().numpy(), G_pred)
            add_from_edge_f_matrix(edge_f_pred_long_data[i].cpu().numpy(), G_pred, node_idx_list)
            G_pred_list.append(G_pred)

    else:
        # generate graphs
        max_num_node = int(args.max_num_node)
        # y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
        # (32, 361, 40). Should be: node_f_pred_long: (BS, Nmax, NF), edge_f_pred_long: (BS, )
        node_f_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_node_feature_num)).cuda()
        #node_f_pred_long[:, :, :] = 1
        edge_f_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node, args.edge_feature_output_dim)).cuda()
        x_step = Variable(torch.ones(test_batch_size,1,args.node_feature_input_dim)).cuda()
        # Rnn should start with all-one input
        # (32, 1, 40)
        for i in range(max_num_node):
            h = rnn(x_step) # Dim: (BS, 1, Hidden)
            # if i == -1:
            #     x_step = Variable(torch.rand(test_batch_size,1,args.node_feature_input_dim)).cuda()
            #     continue
            node_f_pred_step_all = node_f_gen(h) # BS,1,args.max_node_feature_num
            node_f_pred_step = node_f_pred_step_all[:, :, :args.max_node_type_num]
            # node_f_pred_step = torch.softmax(node_f_pred_step, dim=2) # Dim: (BS, 1, Hidden)
            # node_f_pred_step = sample_sigmoid(node_f_pred_step, sample=False, thresh=args.test_thres, sample_time=1)
            # TODO node_f_pred = Mask & node_f_pred

            hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).cuda()
            output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                    dim=0)  # num_layers, batch_size, hidden_size
            # should renew x_step
            x_step = Variable(torch.zeros(test_batch_size,1,args.node_feature_input_dim)).cuda()
            # x_step[:, :, args.max_prev_node:args.max_node_feature_num+args.max_prev_node] = node_f_pred_step

            output_x_step = Variable(torch.ones(test_batch_size,1,args.edge_feature_output_dim)).cuda()
            # Rnn should start with all-one input
            for j in range(min(args.max_prev_node,i+1)):
                output_y_pred_step = output(output_x_step) # (BS, 1, EF)
                # if j == -1:
                #     output_x_step = Variable(torch.rand(test_batch_size,1,args.edge_feature_output_dim)).cuda()
                #     continue
                # output_y_pred_step = torch.softmax(output_y_pred_step, dim=2)
                output_x_step = sample_sigmoid(output_y_pred_step, sample=False, thresh=args.test_thres, sample_time=1,if_sigmoid=True)
                # x_step[:,:,j:j+1] = output_x_step
                edge_f_pred_long[:, i:i + 1, j:j+1, :] = output_x_step.view(output_x_step.size(0), output_x_step.size(1),
                                                                            1, output_x_step.size(2))
                output.hidden = Variable(output.hidden.data).cuda()
                # Add noise to output_x_step
                #output_x_step += Variable(torch.rand(output_x_step.size(0),output_x_step.size(1),output_x_step.size(2))).cuda()
            # y_pred_long[:, i:i + 1, :] = x_step

            
            if args.if_add_test_mask and i != 0:
                child_node_f = torch.matmul(node_f_pred_long[:, :i, :args.max_node_type_num], torch.FloatTensor(args.node_rules).cuda()) # Dim: BS * N' * NF
                indicator_idx = 2 if not args.only_use_adj else 0
                node_f_test_mask = generate_test_mask(edge_f_pred_long[:, i-1, :, indicator_idx], child_node_f, i)
                node_f_pred_step = node_f_test_mask * node_f_pred_step # Dim: BS * 1 * NF
            #elif not args.if_add_test_mask
            if node_f_pred_step.bool().any():
                node_f_pred_step = sample_sigmoid(node_f_pred_step, sample=False, thresh=args.test_thres, sample_time=1)
            else:
                node_f_pred_step = Variable(torch.zeros(node_f_pred_step.size(0),node_f_pred_step.size(1),node_f_pred_step.size(2))).cuda()
                node_f_pred_step[:,:,-1] = 1
            node_f_pred_step_all[:, :, :args.max_node_type_num] = node_f_pred_step
            node_f_pred_step_all[:, :, args.max_node_type_num:] = adjust_node_values(node_f_pred_step_all[:, :, args.max_node_type_num:])
            x_step[:, :, args.max_prev_node:args.max_node_feature_num+args.max_prev_node] = node_f_pred_step_all
            node_f_pred_long[:, i:i+1, :] = node_f_pred_step_all
            node_edge_info = edge_f_pred_long[:, i-1, :, :] # (BS,  M, EF) # where EF = args.edge_feature_output_dim = args.max_edge_feature_num + 2
            indicator_idx = -3 if not args.only_use_adj else -1
            x_step[:, :, :args.max_prev_node] = \
                torch.tensor(torch.tensor(node_edge_info[:, :, indicator_idx:], dtype=torch.bool).any(2), dtype=torch.uint8).\
                    view(test_batch_size, 1, args.max_prev_node) # (BS, 1, M)
            
            if not args.not_use_pooling:
                x_step[:, :, args.max_node_feature_num+args.max_prev_node:] = \
                node_edge_info.mean(dim=1, keepdim=True) # (BS, 1, EF)
            # Add random noise to x_step
            #x_step += Variable(torch.rand(x_step.size(0),x_step.size(1),x_step.size(2))).cuda()
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        # y_pred_long_data = y_pred_long.data.long()
        node_f_pred_long_data = node_f_pred_long.data.float()
        edge_f_pred_long_data = edge_f_pred_long.data.float()

        # save graphs as pickle
        G_pred_list = []
        for i in range(test_batch_size):
            # adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            # G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred = nx.Graph()
            node_idx_list = add_from_node_f_matrix(node_f_pred_long_data[i].cpu().numpy(), G_pred, new_args=args)
            add_from_edge_f_matrix(edge_f_pred_long_data[i].cpu().numpy(), G_pred, node_idx_list)
            G_pred_list.append(G_pred)

    return G_pred_list




def train_rnn_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        output_x = Variable(output_x).cuda()
        output_y = Variable(output_y).cuda()
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())


        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).cuda()
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)
        # print(y_pred.size())
        feature_dim = y_pred.size(0)*y_pred.size(1)
        loss_sum += loss.data[0]*feature_dim/y.size(0)
    return loss_sum/(batch_idx+1)


########### train function for LSTM + VAE
def train(args, dataset_train, rnn, output, node_f_gen=None, edge_f_gen=None, test_set=None):
    # check if load existing model
    if args.load:
        fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname))
        fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
        output.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch<=args.epochs:
        time_start = tm.time()
        # train
        if 'GraphRNN_VAE' in args.note:
            train_vae_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_MLP' in args.note:
            train_mlp_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_RNN' in args.note:
            train_rnn_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output,
                            node_f_gen, edge_f_gen)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
            for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    if 'GraphRNN_VAE' in args.note:
                        G_pred_step = test_vae_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_MLP' in args.note:
                        G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_RNN' in args.note:
                        G_pred_step = test_rnn_epoch(epoch, args, rnn, output, node_f_gen, test_batch_size=args.test_batch_size, test_set=test_set)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                if 'GraphRNN_RNN' in args.note:
                    break
            print('test done, graphs saved')


        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path+args.fname,time_all)


########### for graph completion task
def train_graph_completion(args, dataset_test, rnn, output):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))

    for sample_time in range(1,4):
        if 'GraphRNN_MLP' in args.note:
            G_pred = test_mlp_partial_simple_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        if 'GraphRNN_VAE' in args.note:
            G_pred = test_vae_partial_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        # save graphs
        fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + 'graph_completion.dat'
        save_graph_list(G_pred, fname)
    print('graph completion done, graphs saved')


########### for NLL evaluation
def train_nll(args, dataset_train, dataset_test, rnn, output,graph_validate_len,graph_test_len, max_iter = 1000):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))
    fname_output = args.nll_save_path + args.note + '_' + args.graph_type + '.csv'
    with open(fname_output, 'w+') as f:
        f.write(str(graph_validate_len)+','+str(graph_test_len)+'\n')
        f.write('train,test\n')
        for iter in range(max_iter):
            if 'GraphRNN_MLP' in args.note:
                nll_train = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_test)
            if 'GraphRNN_RNN' in args.note:
                nll_train = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_test)
            print('train',nll_train,'test',nll_test)
            f.write(str(nll_train)+','+str(nll_test)+'\n')

    print('NLL evaluation done')
