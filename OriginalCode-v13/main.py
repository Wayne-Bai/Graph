from train import *

if __name__ == '__main__':
    # All necessary arguments are defined in args.py
    args = Args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    print('File name prefix',args.fname)
    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run"+time, flush_secs=5)

    graphs = create_graphs.create(args)
    
    # split datasets
    random.seed(123)
    shuffle(graphs)
    graphs_train = graphs
    graphs_len = len(graphs)
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_train = graphs#[0:int(0.8*graphs_len)]
    # graphs_test = graphs_train
    # graphs_validate = graphs[0:max(1,int(0.2*graphs_len))]

    # if use pre-saved graphs
    # dir_input = "/dfs/scratch0/jiaxuany0/graphs/"
    # fname_test = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
    #     args.hidden_size_rnn) + '_test_' + str(0) + '.dat'
    # graphs = load_graph_list(fname_test, is_real=True)
    # graphs_test = graphs[int(0.8 * graphs_len):]
    # graphs_train = graphs[0:int(0.8 * graphs_len)]
    # graphs_validate = graphs[int(0.2 * graphs_len):int(0.4 * graphs_len)]


    # graph_validate_len = 0
    # for graph in graphs_validate:
    #     graph_validate_len += graph.number_of_nodes()
    # graph_validate_len /= len(graphs_validate)
    # print('graph_validate_len', graph_validate_len)
    #
    # graph_test_len = 0
    # for graph in graphs_test:
    #     graph_test_len += graph.number_of_nodes()
    # graph_test_len /= len(graphs_test)
    # print('graph_test_len', graph_test_len)



    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    # args.max_num_node = 2000
    # show graphs statistics
    print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge,min_num_edge))
    print('max previous node: {}'.format(args.max_prev_node))

    # save ground truth graphs
    ## To get train and test set, after loading you need to manually slice
    save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

    ### comment when normal training, for graph completion only
    # p = 0.5
    # for graph in graphs_train:
    #     for node in list(graph.nodes()):
    #         # print('node',node)
    #         if np.random.rand()>p:
    #             graph.remove_node(node)
        # for edge in list(graph.edges()):
        #     # print('edge',edge)
        #     if np.random.rand()>p:
        #         graph.remove_edge(edge[0],edge[1])


    ### dataset initialization
    if 'nobfs' in args.note:
        print('nobfs')
        dataset = Graph_sequence_sampler_pytorch_nobfs(graphs_train, max_num_node=args.max_num_node)
        args.max_prev_node = args.max_num_node-1
    if 'barabasi_noise' in args.graph_type:
        print('barabasi_noise')
        dataset = Graph_sequence_sampler_pytorch_canonical(graphs_train,max_prev_node=args.max_prev_node)
        args.max_prev_node = args.max_num_node - 1
    else:
        train_set = Graph_sequence_sampler_pytorch(graphs_train,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
        test_set = Graph_sequence_sampler_pytorch(graphs_test,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(train_set) for i in range(len(train_set))],
                                                                     num_samples=args.batch_size*args.batch_ratio, replacement=True)
    train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                               sampler=sample_strategy)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(test_set) for i in range(len(test_set))],
                                                                     num_samples=args.test_total_size, replacement=True)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                               sampler=sample_strategy)

    ### model initialization
    ## Graph RNN VAE model
    # lstm = LSTM_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_lstm,
    #                   hidden_size=args.hidden_size, num_layers=args.num_layers).cuda()

    if 'GraphRNN_VAE_conditional' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=False).cuda()
        output = MLP_VAE_conditional_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).cuda()
    elif 'GraphRNN_MLP' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=False).cuda()
        output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).cuda()
    elif 'GraphRNN_RNN' in args.note:
        rnn = GRU_plain(input_size=args.node_feature_input_dim, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=True, output_size=args.hidden_size_rnn_output).cuda()
        node_f_gen = MLP_plain(h_size=args.hidden_size_rnn_output, embedding_size=args.embedding_size_output, y_size=args.max_node_feature_num).cuda()
        output = GRU_plain(input_size=args.edge_feature_output_dim, embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                           has_output=True, output_size=args.edge_feature_output_dim).cuda() # TODO: understand input_size, output_size ?
        edge_f_gen = None
        # edge_f_gen = MLP_plain(h_size=args.edge_feature_output_dim, embedding_size=args.embedding_size_output, y_size=args.edge_feature_output_dim).cuda()

    ### start training
    train(args, train_set_loader, rnn, output, node_f_gen, edge_f_gen, test_set=test_set_loader)

    ### graph completion
    # train_graph_completion(args,train_set_loader,rnn,output)

    ### nll evaluation
    # train_nll(args, dataset_loader, dataset_loader, rnn, output, max_iter = 200, graph_validate_len=graph_validate_len,graph_test_len=graph_test_len)

