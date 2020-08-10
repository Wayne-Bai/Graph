
### program configuration

import numpy as np

class Args():
    def __init__(self):
        ### if clean tensorboard
        self.clean_tensorboard = False
        ### Which CUDA GPU device is used for training
        self.cuda = 1

        ### Which GraphRNN model variant is used.
        # The simple version of Graph RNN
        # self.note = 'GraphRNN_MLP'
        # The dependent Bernoulli sequence version of GraphRNN
        self.note = 'GraphRNN_RNN'

        ## for comparison, removing the BFS compoenent
        # self.note = 'GraphRNN_MLP_nobfs'
        # self.note = 'GraphRNN_RNN_nobfs'

        ### Which dataset is used to train the model
        # self.graph_type = 'DD'
        # self.graph_type = 'caveman'
        # self.graph_type = 'caveman_small'
        # self.graph_type = 'caveman_small_single'
        # self.graph_type = 'community4'
        # self.graph_type = 'grid'
        self.graph_type = "AST"
        # self.dataset_type = "2"
        # self.dataset_type = "50"
        self.dataset_type = '50-10'
        # self.dataset_type = "9"
        # self.dataset_type = "50-200"           
        # self.graph_type = 'grid_small'
        # self.graph_type = 'ladder_small'

        # self.graph_type = 'enzymes'
        # self.graph_type = 'enzymes_small'
        # self.graph_type = 'barabasi'
        # self.graph_type = 'barabasi_small'
        # self.graph_type = 'citeseer'
        # self.graph_type = 'citeseer_small'

        # self.graph_type = 'barabasi_noise'
        # self.noise = 10
        #
        # if self.graph_type == 'barabasi_noise':
        #     self.graph_type = self.graph_type+str(self.noise)

        # if none, then auto calculate
        self.max_num_node = None # max number of nodes in a graph
        self.max_prev_node = None # max previous node that looks back

        self.max_node_feature_num = None # max node feature number of desired output # note as NF
        self.max_edge_feature_num = 0 # max edge feature number of desired output
        self.edge_feature_output_dim = None # note as EF
        self.node_feature_input_dim = None # note as INF

        self.not_use_pooling = True

        ### network config
        ## GraphRNN
        if 'small' in self.graph_type:
            self.parameter_shrink = 2
        else:
            self.parameter_shrink = 1
        self.hidden_size_rnn = int(128/self.parameter_shrink) # hidden size for main RNN
        self.hidden_size_rnn_output = 16 # hidden size for output RNN
        self.embedding_size_rnn = int(64/self.parameter_shrink) # the size for LSTM input
        self.embedding_size_rnn_output = 8 # the embedding size for output rnn
        self.embedding_size_output = int(64/self.parameter_shrink) # the embedding size for output (VAE/MLP)

        self.batch_size = 32 # normal: 32, and the rest should be changed accordingly
        self.test_batch_size = 16
        self.test_total_size = 16 # To make things simple, set total_size = batch_size
        self.num_layers = 4

        ### training config
        self.num_workers = 0 # num workers to load data, default 4
        self.batch_ratio = 4 # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
        self.epochs = 5000 # now one epoch means self.batch_ratio x batch_size
        self.epochs_test_start = 2500
        self.epochs_test = 100
        self.epochs_log = 10
        self.epochs_save = 1000

        self.lr = 1e-4
        self.milestones = [300, 800]
        self.lr_rate = 1.0

        self.sample_time = 3 # sample time in each time step, when validating
        self.test_thres = 0.5 # value between 0-1. feature > test_thres => feature:=1, else 0
        # self.node_loss_w = 10.0
        # self.edge_loss_w = 1.5
        # self.loss_type = "mse"
        self.loss_type = "CE"
        self.node_loss_w = 1.5
        self.edge_loss_w = 5

        ### output config
        # self.dir_input = "/dfs/scratch0/jiaxuany0/"
        self.dir_input = "./"
        self.model_save_path = self.dir_input+'model_save/' # only for nll evaluation
        self.graph_save_path = self.dir_input+'graphs/'
        self.figure_save_path = self.dir_input+'figures/'
        self.timing_save_path = self.dir_input+'timing/'
        self.figure_prediction_save_path = self.dir_input+'figures_prediction/'
        self.nll_save_path = self.dir_input+'nll/'


        self.load = False # if load model, default lr is very low
        self.load_epoch = 1000
        self.save = True


        ### baseline config
        # self.generator_baseline = 'Gnp'
        self.generator_baseline = 'BA'

        # self.metric_baseline = 'general'
        # self.metric_baseline = 'degree'
        self.metric_baseline = 'clustering'

        # generate node value matrix
        node_rules = np.zeros((44, 44))
        f = open("nodeRules2matrix.txt", 'r')
        for line in f.readlines():
            line = line.strip('\n')
            row_node, column_node = line.split(' ')
            node_rules[int(row_node) - 1][int(column_node) - 1] = 1

        self.node_rules = node_rules

        self.if_add_train_mask = False
        self.if_add_test_mask = True

        self.if_test_use_groundtruth = False

        ### filenames to save intemediate and final outputs
        self.fname = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_'
        self.fname_pred = self.note+'_'+self.graph_type+self.dataset_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_pred_'
        self.fname_train = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_train_'
        self.fname_test = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_test_'
        self.fname_baseline = self.graph_save_path + self.graph_type + self.generator_baseline+'_'+self.metric_baseline

