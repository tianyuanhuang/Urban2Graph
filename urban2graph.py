import numpy as np
import math

from collections import defaultdict, Counter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

import matplotlib.pyplot as plt
from matplotlib import pylab

import pandas as pd
from tqdm.notebook import tqdm


import torch.optim as optim

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, task='node'):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))

        self.task = task
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.model_type = args.model_type
        self.dropout = args.dropout
        self.num_layers = args.num_layers

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'GAT':
            # When applying GAT with num heads > 1, one needs to modify the 
            # input and output dimension of the conv layers (self.convs),
            # to ensure that the input dim of the next layer is num heads
            # multiplied by the output dim of the previous layer.
            # HINT: In case you want to play with multiheads, you need to change the for-loop when builds up self.convs to be
            # self.convs.append(conv_model(hidden_dim * num_heads, hidden_dim)), 
            # and also the first nn.Linear(hidden_dim * num_heads, hidden_dim) in post-message-passing.
            return GAT

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch


        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
            x = F.dropout(x, p = self.dropout, training = self.training) 
        
        if self.task == "graph":
            x = pyg_nn.global_max_pool(x, batch)

        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)


    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels, reducer='mean', 
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='add')

        self.lin = nn.Linear(in_channels, out_channels) # TODO
        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels) # TODO


        if normalize_embedding:
            self.normalize_emb = True
        else:
            self.normalize_emb = False

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)

    def message(self, x_j, edge_index, size):

        x_j = F.relu(self.lin(x_j)) # TODO
        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]
        
        aggr_out = self.agg_lin( torch.cat( (x, aggr_out), 1 ) )  # TODO
        aggr_out = F.relu(aggr_out)
        if self.normalize_emb:

        return aggr_out


class GAT(pyg_nn.MessagePassing):
    # Please run code with num_heads=1. 
    def __init__(self, in_channels, out_channels, num_heads=1, concat=True,
                 dropout=0, bias=True, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.concat = concat 
        self.dropout = dropout


        self.lin = nn.Linear(num_heads * in_channels, num_heads * out_channels) # TODO


        self.att = nn.Parameter(torch.Tensor(num_heads, num_heads * out_channels * 2 )) # TODO


        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)


    def forward(self, x, edge_index, size=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        x = self.lin(x) # TODO


        # Start propagating messages.
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Constructs messages to node i for each edge (j, i).
        # edge_index_i has shape [E]
        # x_i, x_j has dimension [of edge from i to j]
        

        x_ij = torch.cat((x_i, x_j), 1)
        # x_ij has shape [E, 2 * out_channels]
        # att has shape [2 * out_channels, 1] 
        att = self.att.transpose(0, 1)

        alpha = x_ij @ att
        # alpha has shape [E, 1]
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = pyg_utils.softmax(alpha, edge_index_i) # TODO

        # alpha has shape [E, 1]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        #print(alpha.shape)

        # x_j has shape [E, out_channels]
        # self.heads, 
        return x_j * alpha.view(-1, 1)

    def update(self, aggr_out):
        # Updates node embedings.
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

def train(dataset, task, args):
    if task == 'graph':
        # graph classification: separate dataloader for test set
        data_size = len(dataset)
        loader = DataLoader(
                dataset[:int(data_size * 0.8)], batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(
                dataset[int(data_size * 0.8):], batch_size=args.batch_size, shuffle=True)
    elif task == 'node':
        # use mask to split train/validation/test
        test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        raise RuntimeError('Unknown task')

    # build model
    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, 
                            args, task=task)
    scheduler, opt = build_optimizer(args, model.parameters())

    #print ("loader data", loader)
    #print ("test loader data", test_loader)
    print(len(loader))
    # train
    X = []
    Y = []
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            #print ("batch data", batch)
            #print ("batch train mask", batch.train_mask.sum().item())
            ##print ("batch val mask", batch.val_mask.sum().item())
            #print ("batch test mask", batch.test_mask.sum().item())


            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        print("total_loss = ", total_loss)

        if (epoch + 1) % 3 == 0:
            valid_acc = test(test_loader, model)
            test_acc = test(loader, model, is_validation=False)
            X.append(epoch)
            Y.append(test_acc)
            #Z.append(loader, model)
            print("epoch = {}, valid_acc = {}, test_acc = {}".format(epoch, valid_acc, test_acc))

    # test
    valid_acc = test(test_loader, model, is_validation=True)
    test_acc = test(test_loader, model, is_validation=False)

    print(valid_acc, test_acc,    '   test v/t')
    pylab.rcParams.update({'font.size': 18})
    pylab.figure(figsize=(8, 6))
    pylab.title('{} using {}'.format(args.dataset, args.model_type))
    pylab.xlabel('Epoch')
    pylab.ylabel('Test Accuracy')
    pylab.plot(X, Y, '-o')
    pylab.show()


def test(loader, model, is_validation=True):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
    
    if model.task == 'graph':
        total = len(loader.dataset) 
    else:
        total = 0
        for data in loader.dataset:
            mask = data.val_mask if is_validation else data.test_mask
            total += torch.sum(mask).item()
    return correct / total
  
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def main():
    # Read files
    traj_org = pd.read_csv('./data/20190919_processed.csv')
    user_block = pd.read_csv('./data/user_list_home_block.csv')
    traj_org = traj_org.dropna()
    traj_org = traj_org.reset_index(drop = True)
    user_block = user_block.dropna()
    user_block = user_block.reset_index(drop = True)

    income = pd.read_csv('./data/income_agg_to_blocks.csv')
    income = income.dropna()
    income = income.reset_index(drop = True)

    # Merge into one sheet
    df = pd.merge(user_block, traj_org, on=['id'])
    traj_feat = pd.merge(df, feature, on=['block_id'])
    
    # Map POI_class to POI_ID
    loc_class = traj_org['loc_class'].unique()
    loc_dic = {}
    loc_count = {}
    for i in tqdm(range(len(loc_class))):
        loc_dic[loc_class[i]] = i
        loc_count[loc_class[i]] = 0
    
    def node_to_id(loc_class):
        return loc_dic[loc_class]

    # Translate timestamp to float
    def time_to_float(t):
        rec = ''
        for char in t:
            if char.isdigit():
                rec += char
        rec = float(rec)
        return rec
    
    # Define node features, edges, and edge attributes

    node_list = []
    edge_list = []
    edge_att_list = []

    for block_id in tqdm(all_blocks):
        q = traj_feat[traj_feat['block_id'] == block_id].reset_index(drop = True)
        
        # Define node
        node = []
        # Node features are the total number of check-ins
        loc_count = {}
        for i in range(len(loc_class)):
            loc_count[loc_class[i]] = 0
        count_df = q.groupby(['loc_class']).count().reset_index()
        for i in range(count_df.shape[0]):
            loc_count[count_df['loc_class'][i]] += int(count_df['id'][i])
        for val in loc_count.values():     
            node.append(float(val))
        
        # Define edges
        edge = []
        edge_att = []
        
        if q.shape[0] > 1:
            for i in range(1, q.shape[0]):
                if q['id'][i] == q['id'][i-1] and q['loc_class'][i] != q['loc_class'][i-1]:
                    # Edge represents trajectory between different POI class
                    edge.append([node_to_id(q['loc_class'][i-1]), node_to_id(q['loc_class'][i])])

                    # Edge attributes include [distance, timestamp of start poiint, timestamp of end points]
                    dist = (q['x'][i] - q['x'][i-1])**2 + (q['y'][i] - q['y'][i-1])**2 
                    time_o = time_to_float(q['time'][i-1]) * 10 ** (-10)
                    time_d = time_to_float(q['time'][i]) * 10 ** (-10)
                    edge_att.append([dist * 10000, time_o, time_d])
        
        node_list.append(node)
        edge_list.append(edge)
        edge_att_list.append(edge_att)

    # Define targets to train, need to be 1-dimension LongTensor
    block_id = []
    feat = []
    for i in range(income.shape[0]):
        block_id.append(float(i))
        feat.append([(income['income_under_2499'][i])+\
                 (income['income_2500_3999'][i])*10+\
                 (income['income_4000_7999'][i])*100+\
                 (income['income_above_8000'][i])*1000])

    feature_dic = {}
    for i in range(feature.shape[0]):
        feature_dic[feature['block_id'][i]] = feature['feat'][i]

    y_list = []
    for block_id in all_blocks:
        y_list.append(feature_dic[block_id])

    # Create dataset and call GNNDataset
    data_list = []
    no_edge = []
    for i in tqdm(range(len(y_list))):
        #node = node_list[i]
        node = torch.FloatTensor(node_list_re[i])
        edge = edge_list[i]
        edge_att = edge_att_int_list[i]
        y = y_list[i]
        if edge != []:
            data = Data(x = node,\
                        edge_index = torch.tensor(edge, dtype=torch.long).transpose(0, 1),\
                        edge_attr = torch.tensor(edge_att, dtype=torch.float),\
                        y = torch.tensor(y) )
            data_list.append(data)
        else:
            no_edge.append(i)

    data_l = data_list   # Copy data_list

    class GNNDataset(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None):
            super(GNNDataset, self).__init__(root, transform, pre_transform)
            self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def raw_file_names(self):
            return []
        @property
        def processed_file_names(self):
            return ['./gnn_labels_train.dataset']

        def download(self):
            pass
        
        def process(self):
            
            data_list = data_l
            
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
                
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])


    gnn_dataset = GNNDataset(root='.')

    for args in [
      #{'model_type': 'GCN', 'dataset': 'cora'   , 'num_layers': 2, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.5, 'epochs': 5, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01},
      #{'model_type': 'GraphSage', 'dataset': 'cora'   , 'num_layers': 2, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.5, 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01},
      #{'model_type': 'GAT', 'dataset': 'cora'   , 'num_layers': 2, 'batch_size': 32, 'hidden_dim': 96, 'dropout': 0.3, 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.003},

      {'model_type': 'GraphSage', 'dataset': 'traj_beijing', 'num_layers': 2, 'batch_size': 128, 'hidden_dim': 32, 'dropout': 0.2, 'epochs': 200, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.003},
      #{'model_type': 'GraphSage', 'dataset': 'enzymes', 'num_layers': 2, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.0, 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.001},
      #{'model_type': 'GAT', 'dataset': 'enzymes', 'num_layers': 2, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.0, 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.001},
    ]:
        args = objectview(args)
        if args.dataset == 'traj_beijing':
            dataset = gnn_dataset
            task = 'graph'
            print ("number of graphs", len(dataset))
        
        train(dataset, task, args)


def if __name__ == "__main__":
    main()