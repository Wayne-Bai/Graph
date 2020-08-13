for batch_idx, data in enumerate(data_loader): # Fetch graphs of one batch_size; e.g. 32 graphs
        
        input_node_f_unsorted = data['input_node_f'].float() # Dim: BS * N_max * INF
        raw_node_f_unsorted = data['raw_node_f'].float() # Dim: BS * N_max * NF
        edge_f_unsorted = data['edge_f'].float() # Dim: BS * N_max * M * EF
        y_len_unsorted = data['len'] # list of node numbers in each graph in this batch
        y_len_max = max(y_len_unsorted) # denote as N
        
        input_node_f_unsorted = input_node_f_unsorted[:, 0:y_len_max, :] # Dim: BS * (N+1) * INF
        raw_node_f_unsorted = raw_node_f_unsorted[:, 0:y_len_max, :] # Dim: BS * N * NF
        edge_f_unsorted = edge_f_unsorted[:, 0:y_len_max, :, :] # Dim: BS * N * M * EF
        BS, N, M, EF = edge_f_unsorted.shape
        # initialize GRU hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=input_node_f_unsorted.size(0))
        
        # sort input # The graph with most node numbers come first
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        input_node_f = torch.index_select(input_node_f_unsorted, 0, sort_index)
        raw_node_f = torch.index_select(raw_node_f_unsorted, 0, sort_index)
        edge_f = torch.index_select(edge_f_unsorted, 0, sort_index)

        edge_f_reshape = pack_padded_sequence(edge_f,y_len,batch_first=True).data # SumN * M * EF

        # reverse edge_f_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(edge_f_reshape.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        edge_f_reshape = edge_f_reshape.index_select(0, idx)

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

        output_y = Variable(output_y).cuda() # Dim should be SumN * M * EF

        edge_rnn_input = Variable(edge_rnn_input).cuda()
        input_node_f = Variable(input_node_f).cuda()

        if args.loss_type == "mse":
            output_node_f = Variable(raw_node_f).cuda()
        else:
            output_node_f = Variable(np.argmax(raw_node_f,axis=-1)).cuda()

        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]

        # if using ground truth to test
        h = rnn(input_node_f, pack=True, input_len=y_len) # Dim: BS * (N+1) * hidden_size_rnn_output

        node_f_pred = node_f_gen(h)  # Dim: BS * (N+1) * NF

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
            direction_loss = my_cross_entropy(y_pred[:,:,-4:], torch.argmax(output_y[:,:,-4:],dim=2),if_CE=True)

            # edge_f_loss = my_cross_entropy(y_pred[:,:,:-2], torch.argmax(output_y[:,:,:-2],dim=2))
            edge_f_loss = 0
            node_f_loss = my_cross_entropy(node_f_pred_new, output_node_f,if_CE=True) #+ \
               # binary_cross_entropy_weight(edge_f_pred, output_edge_f)
        loss = args.edge_loss_w * (edge_f_loss + direction_loss) + args.node_loss_w * node_f_loss 
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, node_f_loss: {:.6f}, edge_f_loss: {:.6f}, direction_loss:{:.6f}, num_layer: {}, hidden: {}'.format(
                # epoch, args.epochs,loss.data, node_f_loss.data, edge_f_loss.data, args.num_layers, args.hidden_size_rnn))
                epoch, args.epochs,loss.data, node_f_loss.data, edge_f_loss, direction_loss.data, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.data, epoch*args.batch_ratio+batch_idx)
        feature_dim = N*M
        loss_sum += loss.data*feature_dim
    