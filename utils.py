import time
import random
import torch
import igraph

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import gurobipy as gp
import numpy.random as rand
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.nn as nn
import pyscipopt
import ecole
import torch_geometric



from torch_geometric.nn import GATConv, SAGEConv, ChebConv, GINConv, GCNConv
from torch_geometric.data import Data
# from plotly import graph_objects as go
from gurobipy import GRB
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from copy import deepcopy
# from SetCoverPy import setcover

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_sense_features(graph, weighted = False, directed = False, is_bipartite = False):
    
    if is_bipartite:
        sense_feat_dict = {

            'Degree' : 0,
            'Average Neighbor Degree' : 1,
        }
    
    else: 
        sense_feat_dict = {

            'Degree' : 0,
            'Clustering Coefficient' : 1, 
            'Average Neighbor Degree' : 2,
            'Average Neighbor Clustering' : 3,
            'Node Betweenness' : 4,
            'Structural Holes Constraint' : 5,

        }
    
    ig = igraph.Graph([[e[0], e[1]] for e in nx.to_edgelist(graph)])
    sense_features = np.zeros((len(graph), len(sense_feat_dict)))

    if "Degree" in sense_feat_dict:
        print ("Calculating Degrees...                                   ", end = '\r')
        # Degree
        sense_features[:, sense_feat_dict['Degree']] = list(dict(graph.degree).values())

    if "Average Neighbor Degree" in sense_feat_dict:
        print ("Calculating Average Neighbor Degree...                    ", end = '\r')
        # Neighbor Degree Average
        sense_features[:, sense_feat_dict['Average Neighbor Degree']] = [np.mean([graph.degree[neighbor] for neighbor in dict(graph[node]).keys()]) for node in graph.nodes]

    if "Clustering Coefficient" in sense_feat_dict:
        print ("Calculating Clustering Coefficient...                     ", end = '\r')
        # Clustering Coefficient
        cluster_dict = nx.clustering(graph)
        sense_features[:, sense_feat_dict['Clustering Coefficient']] = list(cluster_dict.values())

    if "Average Neighbor Clustering" in sense_feat_dict:
        print ("Calculating Average Neighbor Clustering Coefficients...   ", end = '\r')
        # Neighbor Average Clustering 
        sense_features[:, sense_feat_dict['Average Neighbor Clustering']] = [np.mean([cluster_dict[neighbor] for neighbor in list(graph[node])]) for node in graph.nodes]
    
    if "Node Betweenness" in sense_feat_dict:
        print ("Calculating Node Betweenness...                           ", end = '\r')
        # Node Betweenness 
        sense_features[:, sense_feat_dict['Node Betweenness']] = ig.betweenness(directed = directed) #list(nx.algorithms.centrality.betweenness_centrality(graph).values())
    
    if "Structural Holes Constraint" in sense_feat_dict:
        print ("Calculating Structural Hole Constraint Scores...         ", end = '\r')
        # Structual Holes
        sense_features[:, sense_feat_dict['Structural Holes Constraint']] = ig.constraint() #list(nx.algorithms.structuralholes.constraint(graph, weight = 'weight').values())
     
    print ("Normalizing Features Between 0 And 1...                   ", end = '\r')
    # Normalise to between 0 and 1 
    sense_features = (sense_features - np.min(sense_features, axis = 0)) / np.ptp(sense_features, axis = 0)
    
    print ("Done                                                      ", end = '\r')
    
    sense_features[np.isnan(sense_features)] = 0
    return sense_feat_dict, sense_features
    
 
def create_graph(universe, subsets):

    graph = nx.Graph()
    hyp_graph = nx.Graph()
            
    # Create Universe Edges 
    for element in universe:
        graph.add_edge(str(universe), str(element))

    
    # Create Subset Edges
    for subset in subsets.values():
        for element in subset:
            graph.add_edge(str(element), str(subset))

            
    hyp_graph = graph.copy().to_undirected()
    hyp_graph.remove_node(str(universe))

    # Select Nodes Reachable From Universe
    reachable = list(nx.bfs_tree(graph, str(universe)))
    subgraph = graph.subgraph(reachable)
    graph = subgraph.copy()   

    # This is the mapping from subsets to graph nodes
    set_to_idx = {n : idx for idx, n in enumerate(graph.nodes)}
    print (len(set_to_idx))
    idx_to_set = {v : k for k, v in set_to_idx.items()}
    graph = nx.relabel_nodes(graph, mapping = set_to_idx, copy = True)
    
    
    
    set_nodes = []
    element_nodes = []
    nodes = list(hyp_graph.nodes)
    hyp_edge_weight = {}
    hyp_node_weight = {}
    
    for idx in range(len(nodes)):
        if type(eval(nodes[idx])) == type(set()):
            set_nodes.append(nodes[idx])
            cover = 0
            for el in eval(nodes[idx]):
                if el in universe: 
                    cover += 1
            hyp_edge_weight[nodes[idx]] = (cover)
        else: 
            element_nodes.append(nodes[idx])
    
    for el in element_nodes:
        hyp_node_weight[el] = (1 / len(hyp_graph[el]))

    hyp_sti = {n : idx for idx, n in enumerate(hyp_graph.nodes)}
    hyp_its = {v : k for k, v in hyp_sti.items()}
    hyp_graph = nx.relabel_nodes(hyp_graph, mapping = hyp_sti, copy = True)
    
    set_nodes_mapped = {x : hyp_sti[x] for x in set_nodes}
    element_nodes_mapped = {x : hyp_sti[x] for x in element_nodes}
    
#     uni_graph = nx.algorithms.bipartite.projected_graph(hyp_graph, [hyp_sti[x] for x in set_nodes])
    
    return_dict = {
        'graph' : graph,
        'hyp_graph' : hyp_graph, 
        'uni_graph' : None,
        'set_to_idx' : set_to_idx,
        'idx_to_set' : idx_to_set,
        'hyp_sti' : hyp_sti, 
        'hyp_its' : hyp_its, 
        'set_nodes_mapped' : set_nodes_mapped,
        'element_nodes_mapped' : element_nodes_mapped,
        'set_nodes' : [hyp_sti[x] for x in set_nodes],
        'element_nodes' : [hyp_sti[x] for x in element_nodes],
        'hyp_edge_weight' : hyp_edge_weight,
        'hyp_node_weight' : hyp_node_weight,    
    }
    
    return return_dict


def calculate_probability_matrix(I, gamma, omega):
    
    # task weight based on current assignment?
    # or should omega here be invariant to assignment?
    
    #omega=np.sum(np.multiply(I, gamma),axis=1)
    R = I * gamma;
    W = I.T * omega;
    
    delta = np.sum(R, axis=1); 
    d = np.sum(W, axis = 1);
    
    P = np.diag(1.0 / d) @ W @ np.diag(1.0 / delta) @ R;
    P = np.nan_to_num(P, 0);
        
    return P;


def current_omega_gamma(I):
    
    '''
    Outputs omega and gamma when given I as the input
    assuming that a uniform distribution of agent energy both across tasks from each agent and across agents from each task
    '''
    I_copy = deepcopy(I)


    budget = I_copy.sum(axis = 0)
    energy = I_copy.sum(axis = 1)

    # H1: edges: tasks, Nodes: agents
    # H2: edges: agents, Nodes: tasks

    omega_H1 = energy
    omega_H2 = budget


    gamma_H1 = I_copy
    gamma_H2 = I_copy.T

    return omega_H1, omega_H2, gamma_H1, gamma_H2



def calculate_Laplacian_matrix (P, k = 1):
    eigenValues, eigenVectors = eigs(P.transpose(), k, which = 'LR');
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    Pi = np.diag(np.abs(eigenVectors[:,0]));

    L = Pi - (Pi@P + P.transpose()@Pi)/2
    
    return L;


def compute_pr(P, r, n, eps=1e-8):
    x = np.ones(n) / n*1.0
    flag = True
    t=0
    while flag:
        x_new = (1 - r) * P @ x
        x_new = x_new + np.ones(n) * r / n
        diff = np.linalg.norm(x_new - x)
        if np.linalg.norm(x_new - x,ord=1) < eps and t > 100:
            flag = False
        t=t+1
        x = x_new
    return x


def get_hyp_features(in_dict, universe, uni = False, hyp = True):

    graph = in_dict['graph'].to_undirected()
    hyp_graph = in_dict['hyp_graph'] 
    uni_graph = in_dict['uni_graph'] 
    set_to_idx = in_dict['set_to_idx'] 
    idx_to_set = in_dict['idx_to_set']
    hyp_sti = in_dict['hyp_sti']
    hyp_its = in_dict['hyp_its'] 
    set_nodes_mapped = in_dict['set_nodes_mapped'] 
    element_nodes_mapped = in_dict['element_nodes_mapped'] 
    hyp_edge_weight = in_dict['hyp_edge_weight'] 
    hyp_node_weight = in_dict['hyp_node_weight']
    set_nodes = in_dict['set_nodes']
    element_nodes = in_dict['element_nodes']
    
    if hyp:
        # Compute Incidence Matrix
        R = np.zeros((len(set_nodes), len(element_nodes)))
        W = np.zeros((len(element_nodes), len(set_nodes)))

        # Generate mapping for hypergraph incidence format
        hyp_element_map = {element_nodes[idx] : idx for idx in range(len(element_nodes))}
        hyp_set_map = {set_nodes[idx] : idx for idx in range(len(set_nodes))}

        # Populate incidence matrix
        incidence_matrix = np.zeros((len(element_nodes), len(set_nodes)))
        for idx in range(len(element_nodes)):

            e_node = hyp_element_map[element_nodes[idx]]
            element_weight = hyp_node_weight[hyp_its[element_nodes[idx]]]

            for set_node in hyp_graph[element_nodes[idx]]:
                subset_weight = hyp_edge_weight[hyp_its[set_node]]
                s_node = hyp_set_map[set_node]
                incidence_matrix[e_node][s_node] = 1
                W[e_node][s_node] = subset_weight

        # Generate Weighted Matrices 
        W_vector = np.zeros((len(hyp_edge_weight), ))

        for sset in hyp_edge_weight:
            W_vector[hyp_set_map[hyp_sti[sset]]] = hyp_edge_weight[sset]

            for el in eval(sset):
                column_index = hyp_element_map[element_nodes_mapped[str(el)]]
                row_index = hyp_set_map[hyp_sti[sset]]
                R[row_index][column_index] = 1 / (sset.count(',') + 1) #hyp_node_weight[str(el)]

        for i in range(R.shape[0]):
            R[i, :] = R[i,:] / sum(R[i,:])

        W = W / W.sum(axis = 1)[:, None]

        # Generate Transition Probability Matrix 
        edge_p = np.transpose(R.dot(W))
        P = np.transpose(W.dot(R))

        # Generate PR Vectors
        subset_pr = compute_pr(edge_p, r = 0.45, n = edge_p.shape[0], eps = 1e-8)
        element_pr = compute_pr(P, r = 0.45, n = P.shape[0], eps = 1e-8)
    
    else: 
        # Generate mapping for hypergraph incidence format
        hyp_element_map = {element_nodes[idx] : idx for idx in range(len(element_nodes))}
        hyp_set_map = {set_nodes[idx] : idx for idx in range(len(set_nodes))}
        subset_pr = np.zeros((len(graph), 1))
        element_pr = np.zeros((len(graph), 1))
        
   
    
    if uni:
        _, uni_sf = get_sense_features(uni_graph, is_bipartite = False, directed = False)
        uni_remap = {n : idx for idx, n in enumerate(uni_graph.nodes)}
        uni_graph = nx.relabel_nodes(uni_graph, uni_remap, copy = True)
    else: 
        uni_sf = np.zeros((len(graph), 6))
        uni_remap = {}
        
    _, graph_sf = get_sense_features(graph, is_bipartite = True, directed = False)
    
    uni_vec = np.zeros((len(graph), 6))
    el_pr_vec = np.zeros((len(graph), 1))
    sub_pr_vec = np.zeros((len(graph), 1))
    cover_vec = np.zeros((len(graph), 1))

    for idx in range(len(graph)):

        node = idx_to_set[idx]

        # Universe Node
        if eval(node) == universe:
            cover_vec[idx] = 0

        # Subset Node
        elif type(eval(node)) == type(set()):
            if uni:
                uni_vec[idx, :] = uni_sf[uni_remap[hyp_sti[node]], :]
            sub_pr_vec[idx, :] = subset_pr[set_nodes.index(hyp_sti[node])]

            sub = eval(node)
            sub_cover = 0
            for elm in sub: 
                if elm in universe:
                    sub_cover = sub_cover + 1
            cover_vec[idx] = sub_cover

        # Element Node
        else:  
            el_pr_vec[idx, :] = element_pr[element_nodes.index(hyp_sti[node])]
            cover_vec[idx] = 0
            
    return graph_sf, uni_vec, el_pr_vec, sub_pr_vec, cover_vec


def train_graph_instance(subsets, universe, solutions, m_type = 'sage', cost_type = 'length', costs = None, epochs = 500, patience = 250, model_path = None, model_name = None, feats = None):
    
            
    if cost_type == 'length':
        optimal_objective = np.sum([len(subsets[int(s)]) for s in solutions[0]])
    elif cost_type == 'equal':
        optimal_objective = np.sum([1 for s in solutions[0]])
    else: 
        optimal_objective = np.sum([costs[int(s)] for s in solutions[0]])
        og_costs = costs.copy()


    return_dict = create_graph(universe = universe,
                               subsets = subsets)

    graph = return_dict['graph'] 
    hyp_graph = return_dict['hyp_graph'] 
    uni_graph = return_dict['uni_graph'] 
    set_to_idx = return_dict['set_to_idx'] 
    idx_to_set = return_dict['idx_to_set']
    hyp_sti = return_dict['hyp_sti']
    hyp_its = return_dict['hyp_its'] 
    set_nodes_mapped = return_dict['set_nodes_mapped'] 
    element_nodes_mapped = return_dict['element_nodes_mapped'] 
    hyp_edge_weight = return_dict['hyp_edge_weight'] 
    hyp_node_weight = return_dict['hyp_node_weight']
    set_nodes = return_dict['set_nodes']
    element_nodes = return_dict['element_nodes']

    if costs is not None:
        subset_to_cost = {str(sorted(v)) : costs[k] for k, v in subsets.items()}
        remapped_costs = {}
        for idx in range(len(graph)):
                
            if str(idx_to_set[idx]) == str(universe):
                remapped_costs[idx] = 0
            
            elif '{' not in str(idx_to_set[idx]):
                remapped_costs[idx] = 0
                
            else: 
                remapped_costs[idx] = subset_to_cost[str(sorted(eval(idx_to_set[idx])))]
                
        
        costs = remapped_costs
         
    
    if cost_type == 'length':
        costs_vec = np.array([len(idx_to_set[node]) for node in graph]).reshape(-1, 1)
    elif cost_type == 'equal':
        costs_vec = np.array([1 for node in graph]).reshape(-1, 1)
    elif cost_type == 'custom':
        costs_vec = np.array(list(costs.values())).reshape(-1, 1)
    
    if cost_type != 'equal':
        costs_vec = (costs_vec - np.min(costs_vec)) / np.ptp(costs_vec)
    
    cat_vec = np.zeros((len(graph), 1))
    universe_node = set_to_idx[str(universe)]
    element_nodes = []
    for node_set, node_id in set_to_idx.items():
        if type(eval(node_set)) == int:
            element_nodes.append(node_id)
            cat_vec[node_id] = 1
    cat_vec[universe_node] = 1
    
    
    # Hypergraph Features
    graph_sf, uni_vec, el_pr_vec, sub_pr_vec, cover_vec = get_hyp_features(return_dict, universe)

    feat_list = []
    if 'bi_sf' in feats:
        feat_list.append(graph_sf)
        
    if 'uni_sf' in feats: 
        feat_list.append(uni_vec)
        
    if 'hyp_feat' in feats:
        feat_list.append(el_pr_vec)
        feat_list.append(sub_pr_vec)
    
    if 'ppr' in feats:
        vec = {idx : 0 for idx in range(len(graph))}
        vec[set_to_idx[str(universe)]] = 1
        ppr_dict = nx.pagerank(graph, personalization = vec)
        ppr_vec = list(dict(sorted(nx.pagerank(graph, personalization = vec).items())).values())
        ppr_vec = np.array(ppr_vec).reshape(-1, 1)
        ppr_vec = (ppr_vec - np.min(ppr_vec, axis = 0)) / np.ptp(ppr_vec, axis = 0)
        feat_list.append(ppr_vec)
        
    if 'costs' in feats:
        feat_list.append(costs_vec)
        
    if 'cat' in feats:
        feat_list.append(cat_vec)
        
    if 'cover' in feats: 
        cover_vec = (cover_vec - np.min(cover_vec, axis = 0)) / np.ptp(cover_vec, axis = 0)
        feat_list.append(cover_vec)
        
    features = np.hstack(feat_list)
    print (features.shape)
        
    edge_index = np.array(list(nx.to_edgelist(graph)))[:, :2].T.astype(int)

    data = Data(x = torch.tensor(features.astype(np.float32)),\
                edge_index = torch.tensor(edge_index))

    data = T.NormalizeFeatures()(data)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    
    y = np.zeros((len(graph), 1))
    
    # Solution sets
    # Build subsets from indices
    for sol in solutions: 
        current_sol = [str(subsets[int(s)]) for s in sol]
        
    # Map subsets to nodes
    for sol in current_sol:
        node = set_to_idx[str(sol)]
        y[node] = 1
        
    # Element nodes  
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Class labels
    _, counts = np.unique(y, return_counts = True)
    class_weights = 1 - (counts / np.sum(counts))
    weight = [class_weights[int(y[idx])] for idx in range(len(y))]
    print ("Class Counts: ", class_weights)
    print ("Graph Size: ", len(graph))
    y = torch.tensor(y)
    y = y.type(torch.float32)
    y = y.to(device)
    weight = torch.tensor(weight).reshape(-1, 1)# + 1
    weight = weight.type(torch.float32)
    weight = weight.to(device)
    
    # Create model
    if m_type == 'gat':
        model = GAT(hidden = 32, 
                      in_head = 16, 
                      out_head = 64, 
                      num_features = data.x.shape[1], 
                      out_dim = 64)
        
    elif m_type == 'sage':
        model = SAGE(in_channels = data.x.shape[1],
                 hidden_channels = 512,
                 out_channels = 512,
                 num_layers = 2,
                 dropout = 0.4).to(device)
        
    elif m_type == 'gin':
        model = GIN(in_channels = data.x.shape[1],
                 hidden_channels = 512,
                 out_channels = 512,
                 num_layers = 2,
                 dropout = 0.4).to(device)
        
    elif m_type == 'gcn':
        model = GCN(in_channels = data.x.shape[1],
                 hidden_channels = 512,
                 out_channels = 512,
                 num_layers = 2,
                 dropout = 0.4).to(device)
        
    elif m_type == 'cheb':
        model = Cheb(in_channels = data.x.shape[1],
                 hidden_channels = 512,
                 out_channels = 512,
                 num_layers = 2,
                 dropout = 0.4).to(device)

    model.to(device)
    model.optimizer.zero_grad()

    if model_path is not None:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else: 
            model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
           
    model.optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 5e-4)
    model.train()

    # Train
    best = 1e9
    wait = 0

    for ep in range(epochs):
        print ('Epoch: ', str(ep), end = '\r')
        model.optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.binary_cross_entropy(out, y)
        if loss < best:
            best = loss
            wait = 0

            torch.save(model.state_dict(), './' + model_name + '.pkl')
        else:
            wait = wait + 1

        if wait == patience: 
            print ("Early Stopping")
            break

        loss.backward()
        model.optimizer.step()
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('./' + model_name + '.pkl'))
    else: 
        model.load_state_dict(torch.load('./' + model_name + '.pkl', map_location = torch.device('cpu')))
        
    
    # Evaluation
    model.eval()
    out = model(x, edge_index)
    np_out = out.cpu().detach().numpy().reshape(-1, )  
    y = y.cpu().detach().numpy()

    # Find Threshold 
    thresholds = np.arange(0, 1.02, 0.02)
    f1_t_list = []
    for t in thresholds: 
        f1_t_list.append(f1_score(y, (np_out > t).astype(int)))

    f1 = max(f1_t_list)
    threshold = thresholds[np.argmax(f1_t_list)]

    auc = roc_auc_score(y, np_out)
    aup = average_precision_score(y, np_out)
    acc = accuracy_score(y, (np_out > threshold).astype(int))
    conf = confusion_matrix(y, (np_out > threshold).astype(int))

    print ("F1 : ", f1)
    print ("AUC : ", auc)
    print ("AUP :", aup)
    print ("Acc : ", acc)
    
    del model

    print ("Output:")
    plt.hist(np_out)
    plt.show()

    return f1, auc, aup, acc, np.divide(*class_weights)


def train_set_cover(train_instances, cost_type, epochs, patience, model_path, model_name, feats, m_type):
    
    f1_list = []
    auc_list = []
    aup_list = []
    acc_list = []
    ratio_list = []
    
    
    for idx in range(len(train_instances)):
        print ("Instance : ", idx, "/", len(train_instances))
        if train_instances[idx]['solutions'] != []:
            
            if cost_type == 'custom':
                costs = train_instances[idx]['costs']
            else: 
                costs = None
            

            f1, auc, aup, acc, ratio = train_graph_instance(subsets = train_instances[idx]['subsets'],
                                                     universe = train_instances[idx]['universe'],
                                                     solutions = train_instances[idx]['solutions'],
                                                     cost_type = cost_type,
                                                     costs = costs,
                                                     epochs = epochs,
                                                     patience = patience,
                                                     model_path = model_path,
                                                     model_name = model_name, 
                                                     feats = feats, 
                                                     m_type = m_type)

            f1_list.append(f1)
            auc_list.append(auc)
            aup_list.append(aup)
            acc_list.append(acc)
            ratio_list.append(ratio)

            if model_path is None:
                model_path = './' + model_name + '.pkl'

            
    return f1_list, auc_list, aup_list, auc_list, ratio_list


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))
        
        self.final = nn.Linear(out_channels, 1)
        self.dropout = dropout
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 9e-5, weight_decay = 5e-4)
        
        self.return_inter = False

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.final.reset_parameters()

    def forward(self, x, adj_t):
        for bn, conv in zip(self.bns[:-1], self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = bn(x)
        x = self.convs[-1](x, adj_t)
        x = self.bns[-1](x)
        inter = x
        
        x = self.final(x)
        x = torch.sigmoid(x)
        
        if self.return_inter:
            return x, inter
        else:
            return x
        
class Cheb(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(Cheb, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(ChebConv(in_channels, hidden_channels, K = 2))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(ChebConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(ChebConv(hidden_channels, out_channels, K = 2))
        self.bns.append(nn.BatchNorm1d(out_channels))
        
        self.final = nn.Linear(out_channels, 1)
        self.dropout = dropout
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 9e-5, weight_decay = 5e-4)
        
        self.return_inter = False

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.final.reset_parameters()

    def forward(self, x, adj_t):
        for bn, conv in zip(self.bns[:-1], self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = bn(x)
        x = self.convs[-1](x, adj_t)
        x = self.bns[-1](x)
        inter = x
        
        x = self.final(x)
        x = torch.sigmoid(x)
        
        if self.return_inter:
            return x, inter
        else:
            return x
    
    
class GAT(torch.nn.Module):
    
    def __init__(self, hidden, in_head, out_head, num_features, out_dim):
        super(GAT, self).__init__()
        
        self.hidden = hidden
        self.in_head = in_head
        self.out_head = out_head
        self.num_features = num_features
        self.out_dim = out_dim
        self.return_inter = False
        
        self.conv1 = GATConv(self.num_features,
                             self.hidden,
                             heads = self.in_head,
                             dropout = 0.6)
        
        self.conv2 = GATConv(self.hidden * self.in_head,
                             self.out_dim,
                             concat = False,
                             heads = self.out_head,
                             dropout = 0.6)
        
        self.linear_one = nn.Linear(self.out_dim, self.out_dim)
        self.linear_two = nn.Linear(self.out_dim, self.out_dim)
        self.final = nn.Linear(self.out_dim, 1)
        
        self.bn_conv_one = nn.BatchNorm1d(self.hidden * self.in_head)
        self.bn_conv_two = nn.BatchNorm1d(self.out_dim)
        self.bn_one = nn.BatchNorm1d(self.out_dim)
        self.bn_two = nn.BatchNorm1d(self.out_dim)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 9e-5, weight_decay = 5e-4)
                
        
    def forward(self, x, edge_index):
            
            x = self.conv1(x, edge_index)
            x = self.bn_conv_one(x)
            x = F.relu(x)
            x = F.dropout(x, p = 0.6,
                          training = self.training)
            x = self.conv2(x, edge_index)
            x = self.bn_conv_two(x)
            x = F.dropout(x,
                          p = 0.6,
                          training = self.training)
            
            x = self.linear_one(x)
            x = self.bn_one(x)
            x = F.relu(x)
            x = F.dropout(x,
                          p = 0.6,
                          training = self.training)
            
            x = self.linear_two(x)
            x = self.bn_two(x)
            x = F.relu(x)
            inter = x
            x = F.dropout(x,
                          p = 0.5,
                          training = self.training)

            x = self.final(x)
            x = torch.sigmoid(x)
            
            if self.return_inter:
                return x, inter
            else:
                return x
        
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
                
        self.convs.append(GINConv(nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                                nn.ReLU(),
                                                nn.BatchNorm1d(hidden_channels),
                                               ), eps=0., train_eps = False))
        
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                                nn.ReLU(),
                                                nn.BatchNorm1d(hidden_channels),
                                               ), eps=0., train_eps = False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_channels, out_channels),
                                                nn.ReLU(),
                                                nn.BatchNorm1d(hidden_channels),
                                               ), eps=0., train_eps = False))
        self.bns.append(nn.BatchNorm1d(out_channels))
        
        self.final = nn.Linear(out_channels, 1)
        self.dropout = dropout
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 9e-5, weight_decay = 5e-4)
        
        self.return_inter = False

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.final.reset_parameters()

    def forward(self, x, adj_t):
        for bn, conv in zip(self.bns[:-1], self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = bn(x)
        x = self.convs[-1](x, adj_t)
        x = self.bns[-1](x)
        inter = x
        
        x = self.final(x)
        x = torch.sigmoid(x)
        
        if self.return_inter:
            return x, inter
        else:
            return x
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))
        
        self.final = nn.Linear(out_channels, 1)
        self.dropout = dropout
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 9e-5, weight_decay = 5e-4)
        
        self.return_inter = False

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.final.reset_parameters()

    def forward(self, x, adj_t):
        for bn, conv in zip(self.bns[:-1], self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = bn(x)
        x = self.convs[-1](x, adj_t)
        x = self.bns[-1](x)
        inter = x
        
        x = self.final(x)
        x = torch.sigmoid(x)
        
        if self.return_inter:
            return x, inter
        else:
            return x
    
    
def gurobi_solver(subsets, universe, cost_type = 'equal', costs = None, initial_values = None, time_limit = None, stop_obj = None, callback = False, return_model_only = False):

    
    # Create a new model
    m = gp.Model("set_cover")

    # Create variables
    x = m.addVars(list(subsets.keys()),
                  vtype = GRB.BINARY,
                  name = [str(n) for n in subsets.keys()])
    
    if cost_type == 'length':
        costs = {key : len(subsets[key]) for key in subsets.keys()}
    elif cost_type == 'equal':
        costs = {key : 1 for key in subsets.keys()}
    elif cost_type == 'custom':
        None
    else: 
        print ("Invalid Cost Type")
        return
        
    
    # Set objective
    m.setObjective(gp.quicksum([costs[i] * x[i] for i in subsets.keys()]), GRB.MINIMIZE)

    # Add constraints
    for j in universe:
        m.addConstr(gp.quicksum(x[i] for i in subsets.keys() if j in subsets[i]) >= 1)

    m.setParam('Presolve', 0)    
    if time_limit is not None: 
        m.setParam('TimeLimit', time_limit)
        
    if stop_obj is not None: 
        m.setParam('BestObjStop', stop_obj)
        
    if initial_values is not None: 
        for i in range(len(subsets.keys())):
            x[i].start = initial_values[i]
    
    if return_model_only:
        return m
    
    # Optimize model
    gurobi_time_start = time.time()
    if callback:
        m.optimize(callback_fn)
    else:
        m.optimize()
    gurobi_time_end = time.time()
    
    if m.status == gp.GRB.TIME_LIMIT and m.solCount == 0:
        return None, None


    # Print solution
    solutions = []
    gurobi_time = np.round(gurobi_time_end - gurobi_time_start, 4)
    if 1:#m.status == gp.GRB.OPTIMAL:
        for i in range(m.SolCount):
            current_sol = []
            m.setParam(gp.GRB.Param.SolutionNumber, i)
            for v in m.getVars():
                if v.x > 0.5:
                    current_sol.append(v.varName)
            solutions.append(current_sol)

    return solutions, gurobi_time, m.NodeCount

def callback_fn(model, where):
    
    if where == GRB.Callback.MIP:
        
        obj = int(model.cbGet(GRB.Callback.MIP_OBJBST))
        t = model.cbGet(GRB.Callback.RUNTIME)
        
        if obj not in obj_time: 
            obj_time[obj] = t
        
        
def greedy_approx(universe, subsets, weights = None):
    
    start_time = time.time()
    U = universe
    R = U
    S = list(subsets.values())
    
    if weights is not None:
        w = weights 
    else:
        w = [len(x) for x in S]

    C = []
    costs = []
    
    def findMin(S, R):
        minCost = 99999.0
        minElement = -1
        for i, s in enumerate(S):
            try:
                cost = w[i]/(len(s.intersection(R)))
                if cost < minCost:
                    minCost = cost
                    minElement = i
            except:
                # Division by zero, ignore
                pass
        return S[minElement], w[minElement]

    while len(R) != 0:
        S_i, cost = findMin(S, R)
        C.append(S_i)
        R = R.difference(S_i)
        costs.append(cost)
        
    run_time = time.time() - start_time
    return C, sum(costs), run_time
obj_time = {}   


def test_graph_instance(subsets, universe, solutions, m_type = 'sage', cost_type = 'length', costs = None,
                        model_path = None, feats = None, dec_size = 20,
                        perc_start = 60, init_sol_perc = 90, obj_threshold = 0.9, time_limit = 100, 
                        break_intermediate = False, 
                        neurons = None, run_random = True, 
                        gnn_solver = 'gurobi'):
    
    global obj_time
    obj_time = {}
    
    if not break_intermediate:
        
        m = gurobi_solver(subsets = subsets,
                           universe = universe,
                           cost_type = cost_type,
                           costs = costs,
                           time_limit = time_limit, 
                           callback = True, 
                          return_model_only = True)
        
        mat = m.getA().todense()
        
        if gnn_solver == 'gurobi':
            solutions, gurobi_time, gu_nc = gurobi_solver(subsets = subsets,
                                                   universe = universe,
                                                   cost_type = cost_type,
                                                   costs = costs,
                                                   time_limit = time_limit, 
                                                   callback = True)
        elif gnn_solver == 'scip':
            
            if cost_type == 'equal':
                costs = {}
                for k in subsets:
                    costs[k] = 1
            
            solutions, gurobi_time, gu_nc, _ = scip_solve(universe = [i for i in range(mat.shape[0])],
                                                                       subsets = [list(x) for x in subsets.values()],
                                                                       costs = list(costs.values()), 
                                                                       init_sets = None)
        
        
        
    if cost_type == 'length':
        optimal_objective = np.sum([len(subsets[int(s)]) for s in solutions[0]])
    elif cost_type == 'equal':
        optimal_objective = np.sum([1 for s in solutions[0]])
    else: 
        optimal_objective = np.sum([costs[int(s)] for s in solutions[0]])
        og_costs = costs.copy()
    
    return_dict = create_graph(universe = universe,
                               subsets = subsets)

    graph = return_dict['graph'] 
    hyp_graph = return_dict['hyp_graph'] 
    uni_graph = return_dict['uni_graph'] 
    set_to_idx = return_dict['set_to_idx'] 
    idx_to_set = return_dict['idx_to_set']
    hyp_sti = return_dict['hyp_sti']
    hyp_its = return_dict['hyp_its'] 
    set_nodes_mapped = return_dict['set_nodes_mapped'] 
    element_nodes_mapped = return_dict['element_nodes_mapped'] 
    hyp_edge_weight = return_dict['hyp_edge_weight'] 
    hyp_node_weight = return_dict['hyp_node_weight']
    set_nodes = return_dict['set_nodes']
    element_nodes = return_dict['element_nodes']
    

    if costs is not None:
        subset_to_cost = {str(sorted(v)) : costs[k] for k, v in subsets.items()}
        remapped_costs = {}
        for idx in range(len(graph)):
                
            if str(idx_to_set[idx]) == str(universe):
                remapped_costs[idx] = 0
            
            elif '{' not in str(idx_to_set[idx]):
                remapped_costs[idx] = 0
                
            else: 
                remapped_costs[idx] = subset_to_cost[str(sorted(eval(idx_to_set[idx])))]
                
        
        costs = remapped_costs
         
    
    if cost_type == 'length':
        costs_vec = np.array([len(idx_to_set[node]) for node in graph]).reshape(-1, 1)
    elif cost_type == 'equal':
        costs_vec = np.array([1 for node in graph]).reshape(-1, 1)
    elif cost_type == 'custom':
        costs_vec = np.array(list(costs.values())).reshape(-1, 1)
    
    if cost_type != 'equal':
        costs_vec = (costs_vec - np.min(costs_vec)) / np.ptp(costs_vec)
    
    cat_vec = np.zeros((len(graph), 1))
    universe_node = set_to_idx[str(universe)]
    element_nodes = []
    for node_set, node_id in set_to_idx.items():
        if type(eval(node_set)) == int:
            element_nodes.append(node_id)
            cat_vec[node_id] = 1
    cat_vec[universe_node] = 1
    
    
    # Hypergraph Features
    if 'hyp_feat' not in feats:
        graph_sf, uni_vec, el_pr_vec, sub_pr_vec, cover_vec = get_hyp_features(return_dict, universe, hyp = False)
    else: 
        graph_sf, uni_vec, el_pr_vec, sub_pr_vec, cover_vec = get_hyp_features(return_dict, universe)
        
    feat_list = []
    if 'bi_sf' in feats:
        feat_list.append(graph_sf)
        
    if 'uni_sf' in feats: 
        feat_list.append(uni_vec)
        
    if 'hyp_feat' in feats:
        feat_list.append(el_pr_vec)
        feat_list.append(sub_pr_vec)
    
    if 'ppr' in feats:
        vec = {idx : 0 for idx in range(len(graph))}
        vec[set_to_idx[str(universe)]] = 1
        ppr_dict = nx.pagerank(graph, personalization = vec)
        ppr_vec = list(dict(sorted(nx.pagerank(graph, personalization = vec).items())).values())
        ppr_vec = np.array(ppr_vec).reshape(-1, 1)
        ppr_vec = (ppr_vec - np.min(ppr_vec, axis = 0)) / np.ptp(ppr_vec, axis = 0)
        feat_list.append(ppr_vec)
        
    if 'costs' in feats:
        feat_list.append(costs_vec)
        
    if 'cat' in feats:
        feat_list.append(cat_vec)
        
    if 'cover' in feats: 
        cover_vec = (cover_vec - np.min(cover_vec, axis = 0)) / np.ptp(cover_vec, axis = 0)
        feat_list.append(cover_vec)
        
    features = np.hstack(feat_list)
        
    #### GNN Model ####
    pred_start_time = time.time()
    
    edge_index = np.array(list(nx.to_edgelist(graph)))[:, :2].T.astype(int)

    data = Data(x = torch.tensor(features.astype(np.float32)),\
                edge_index = torch.tensor(edge_index))

#     data = T.NormalizeFeatures()(data)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    
    y = np.zeros((len(graph), 1))
    
    # Solution sets
    # Build subsets from indices
    for sol in solutions: 
        current_sol = [str(subsets[int(s)]) for s in sol]
    
    if torch.cuda.is_available:
        torch.cuda.empty_cache()
    
    # Create model
    if m_type == 'gat':
        model = GAT(hidden = 32, 
                      in_head = 16, 
                      out_head = 64, 
                      num_features = data.x.shape[1], 
                      out_dim = 64)
        
    elif m_type == 'sage':
        if neurons is None:
            model = SAGE(in_channels = data.x.shape[1],
                     hidden_channels = 512,
                     out_channels = 512,
                     num_layers = 2,
                     dropout = 0.4).to(device)
        else: 
            model = SAGE(in_channels = data.x.shape[1],
                     hidden_channels = neurons,
                     out_channels = neurons,
                     num_layers = 2,
                     dropout = 0.4).to(device)
            
    elif m_type == 'gin':
        if neurons is None:
            model = GIN(in_channels = data.x.shape[1],
                     hidden_channels = 512,
                     out_channels = 512,
                     num_layers = 2,
                     dropout = 0.4).to(device)
        else: 
            model = GIN(in_channels = data.x.shape[1],
                     hidden_channels = neurons,
                     out_channels = neurons,
                     num_layers = 2,
                     dropout = 0.4).to(device)
            
    elif m_type == 'gcn':
        if neurons is None:
            model = GCN(in_channels = data.x.shape[1],
                     hidden_channels = 512,
                     out_channels = 512,
                     num_layers = 2,
                     dropout = 0.4).to(device)
        else: 
            model = GCN(in_channels = data.x.shape[1],
                     hidden_channels = neurons,
                     out_channels = neurons,
                     num_layers = 2,
                     dropout = 0.4).to(device)
            
    elif m_type == 'cheb':
        if neurons is None:
            model = Cheb(in_channels = data.x.shape[1],
                     hidden_channels = 512,
                     out_channels = 512,
                     num_layers = 2,
                     dropout = 0.4).to(device)
        else: 
            model = Cheb(in_channels = data.x.shape[1],
                     hidden_channels = neurons,
                     out_channels = neurons,
                     num_layers = 2,
                     dropout = 0.4).to(device)
        
    model.to(device)
    model.optimizer.zero_grad()

    model.load_state_dict(torch.load(model_path, map_location = device))
       

    # Evaluation
    model.eval()
    out = model(x, edge_index)
    out_np = out.cpu().detach().numpy().reshape(-1, )  
    # plt.hist(out_np, bins = 50)
    # plt.show()
    sol_nodes = []
    universe_node = set_to_idx[str(universe)]
    element_nodes = []
    for node_set, node_id in set_to_idx.items():
        if type(eval(node_set)) == int:
            element_nodes.append(node_id)
            
    if break_intermediate:
        return x, edge_index, set_to_idx, out_np
#     out_np[element_nodes] = 0
    out_np = np.array(list(zip(range(len(graph)), out_np)))        
    out_np = out_np[out_np[:, 1].argsort()][::-1]  
    perc = perc_start
    sub_nodes = list(out_np[:, 0][np.where(out_np[:, 1] > np.percentile(out_np[:, 1], perc))[0]])
    init_sol_nodes = list(out_np[:, 0][np.where(out_np[:, 1] > np.percentile(out_np[:, 1], init_sol_perc))[0]])
    
    sols = []
    run_count = 0
    current_objective = np.inf
    while sub_nodes == []:
        perc = max(0, perc - dec_size)
        sub_nodes = list(out_np[:, 0][np.where(out_np[:, 1] > np.percentile(out_np[:, 1], perc))[0]])
        
    while init_sol_nodes == []:
        init_sol_perc = max(0, init_sol_perc - dec_size)
        init_sol_nodes = list(out_np[:, 0][np.where(out_np[:, 1] > np.percentile(out_np[:, 1], init_sol_perc))[0]])
    
    if len(sub_nodes) == len(subsets):
        skip_flag = True
    else: 
        skip_flag = False
    solve_start = time.time()
    
    while (optimal_objective / current_objective) < obj_threshold:
        
        if skip_flag is False:
            pred_sets = [eval(idx_to_set[n]) for n in sub_nodes]
            init_sol_sets = [eval(idx_to_set[n]) for n in init_sol_nodes]
            init_sets = {}
            preds = {}
            if costs is not None:
                sub_costs = {}
            else:
                sub_costs = None
            counter = 0
            
            for idx in range(len(pred_sets)):
                if type(pred_sets[idx]) == type(set()):
                    # Make sure universe is not part of subsets and nor are element nodes
                    if pred_sets[idx] != universe and len(pred_sets[idx]) != 1:

                        if costs is not None:
                            sub_costs[counter] = subset_to_cost[str(sorted(pred_sets[idx]))]

                        if pred_sets[idx] in init_sol_sets:
                            init_sets[counter] = pred_sets[idx]
                        preds[counter] = pred_sets[idx]
                        counter = counter + 1

        else: 
            break
            preds = subsets
            sub_costs = og_costs
            
        if perc == 0:
            preds = subsets
            
        
        if init_sol_perc == 100:
            init_values = None
        else:
            init_values = np.zeros(len(preds))
            for idx, sub in preds.items():
                if sub in init_sol_sets:
                    init_values[idx] = 1
        
        obj_time_gu = obj_time.copy()
        obj_time = {}
        
        if run_count > 0 and warm_start is not None:
            init_sets = {}
            init_values = np.zeros(len(preds))
            for idx, sub in preds.items():
                if sub in warm_start:
                    init_values[idx] = 1
                    init_sets[idx] = sub
                
            
        if gnn_solver == 'gurobi':
            sols, gu_time, gnn_nc = gurobi_solver(subsets = preds,
                                          universe = universe,
                                          cost_type = cost_type,
                                          costs = sub_costs,
                                          time_limit = time_limit, 
                                          initial_values = init_values, 
                                          callback = True)
        elif gnn_solver == 'scip':
            try:
                sols, gu_time, gnn_nc, _ = scip_solve(universe = [i for i in range(mat.shape[0])],
                                                   subsets = [list(x) for x in preds.values()],
                                                   costs = list(sub_costs.values()), 
                                                   init_sets = init_sets)
            except:
                sols = []
        else: 
            # Default to Gurobi
            sols, gu_time, gnn_nc = gurobi_solver(subsets = preds,
                                          universe = universe,
                                          cost_type = cost_type,
                                          costs = sub_costs,
                                          time_limit = time_limit, 
                                          initial_values = init_values, 
                                          callback = True)


        if len(sols) > 0:
            warm_start = [preds[int(s)] for s in sols[0]]
        else: 
            warm_start = None
        
        if sols != []:
            
            if cost_type == 'length':
                current_objective = np.sum([len(preds[int(s)]) for s in sols[0]])
            elif cost_type == 'equal':
                current_objective = np.sum([1 for s in sols[0]])
            elif cost_type == 'custom': 
                current_objective = np.sum([sub_costs[int(s)] for s in sols[0]])
            
            
        perc = max(0, perc - dec_size)
        sub_nodes = list(out_np[:, 0][np.where(out_np[:, 1] > np.percentile(out_np[:, 1], perc))[0]])
    
        if perc == 0:
            break
            
        run_count = run_count + 1
        
        
    if skip_flag:
        solve_time = gurobi_time
        current_objective = optimal_objective
        preds = subsets
    else:
        solve_time = time.time() - solve_start
    
    if run_random:
        print ("~~~~~~~~~~ RANDOM ~~~~~~~~~~~")

        non_zero = np.where(out_np != 0)[0].shape[0]
        r_res = {}
        for r_perc in [0.2, 0.5, 0.8]:

            random_start = time.time()

            r_sol = []
            r_idx = int(r_perc * non_zero)
            current_objective_r = np.inf
            random_sets = np.random.permutation(out_np[:non_zero, 0])
            r_sub = random_sets[:r_idx]

            pred_sets_r = [eval(idx_to_set[n]) for n in r_sub]

            preds_r = {}
            if costs is not None:
                sub_costs_r = {}
            else:
                sub_costs_r = None



            counter = 0
            for idx in range(len(pred_sets_r)):
                if type(pred_sets_r[idx]) == type(set()):
                    # Make sure universe is not part of subsets and nor are element nodes
                    if pred_sets_r[idx] != universe and len(pred_sets_r[idx]) != 1:

                        if costs is not None:
                            sub_costs_r[counter] = subset_to_cost[str(sorted(pred_sets_r[idx]))]

                        preds_r[counter] = pred_sets_r[idx]
                        counter = counter + 1

            if r_perc == 1:
                preds_r = subsets
                if costs is not None: 
                    sub_costs_r = og_costs

            r_sol, _, _ = gurobi_solver(subsets = preds_r,
                                          universe = universe,
                                          cost_type = cost_type,
                                          costs = sub_costs_r,
                                          time_limit = 2000)
            if r_sol != []:

                if cost_type == 'length':
                    current_objective_r = np.sum([len(preds_r[int(s)]) for s in r_sol[0]])
                elif cost_type == 'equal':
                    current_objective_r = np.sum([1 for s in r_sol[0]])
                else: 
                    current_objective_r = np.sum([sub_costs_r[int(s)] for s in r_sol[0]])

            #r_perc = min(1, r_perc + 0.3)
            r_idx = int(r_perc * len(graph))
            r_sub = random_sets[:r_idx]

            random_time = time.time() - random_start
            r_res[r_perc] = {}
            r_res[r_perc]['time'] = random_time
            r_res[r_perc]['objective'] = current_objective_r
    else: 
        r_res = {0.2 : {'time' : 0, 'objective' : 0},
                 0.5: {'time' : 0, 'objective' : 0},
                 0.8 : {'time' : 0, 'objective' : 0}}
        preds_r = []
        
    
    
    if cost_type == 'custom':
        greedy_sols, greedy_obj, greedy_time = greedy_approx(universe, subsets, weights = list(og_costs.values()))
    else:
        greedy_sols, greedy_obj, greedy_time = greedy_approx(universe, subsets)
    
    greedy_sols_model, greedy_obj_model, greedy_time_model = greedy_approx(universe, preds)
    
    if current_objective > optimal_objective:
        try:
            gur_eq_time = obj_time_gu[current_objective]
        except:
            gur_eq_time = gurobi_time #obj_time_gu[current_objective - 1]
            
        try:
            gnn_eq_time = obj_time[current_objective]
        except: 
            try:
                gnn_eq_time = obj_time[current_objective - 1]
            except: 
                gnn_eq_time = solve_time
            
    else: 
        gur_eq_time = gurobi_time
        gnn_eq_time = solve_time
    
    
    print ("#########################################")
    
    if gnn_solver == 'scip':
        print_m = 'SCIP'
    else: 
        print_m = 'Gurobi'
        
    print ("Graph-SCP Time: ", solve_time)
    print ("Random Time:", r_res[0.2]['time'], r_res[0.5]['time'], r_res[0.8]['time'])
    print (print_m, "Time:", gurobi_time)
    print ("Greedy Time:", greedy_time)
    print ()
    print ("Optimal Objective: ", optimal_objective)
    print (print_m, "Objective: ", current_objective)
    print ("Random Objective: ", r_res[0.2]['objective'], r_res[0.5]['objective'], r_res[0.8]['objective'])
    print ()
    print ("Gurobi Node Count : ", gu_nc)
    print (print_m, "Node Count : ", gnn_nc)
    print ()
    print ("Num Times Gurobi Run : ", run_count)
    print ("Problem Size : ", len(preds), "/", len(subsets))
    print ("#########################################")
    

    return [optimal_objective, current_objective, r_res[0.2]['objective'], r_res[0.5]['objective'], r_res[0.8]['objective'], greedy_obj, greedy_obj_model,
            solve_time, r_res[0.2]['time'], r_res[0.5]['time'], r_res[0.8]['time'], gurobi_time, greedy_time, greedy_time_model,
            len(preds), len(preds_r), len(subsets), gu_nc, gnn_nc, run_count], preds



def test_set_cover(test_instances, cost_type, num_runs, model_path, feats, dec_size, obj_threshold, perc_start, init_sol_perc, m_type, time_limit, run_random, gnn_solver, neurons):
    
    results = []
    instances = []
    
    if test_instances is not None: 
        for idx in range(len(test_instances)):
            if cost_type == 'custom':
                costs = test_instances[idx]['costs']
            else: 
                costs = None
                
            return_list, instances = test_graph_instance(subsets = test_instances[idx]['subsets'],
                                                universe = test_instances[idx]['universe'],
                                                solutions = test_instances[idx]['solutions'],
                                                cost_type = cost_type,
                                                costs = costs,
                                                model_path = model_path,
                                                feats = feats,
                                                dec_size = dec_size, 
                                                perc_start = perc_start,
                                                init_sol_perc = init_sol_perc,
                                                obj_threshold = obj_threshold, 
                                                m_type = m_type, 
                                                time_limit = time_limit, 
                                                run_random = run_random, 
                                                gnn_solver = gnn_solver,
                                                neurons = neurons)
            print ("################################")
            print ("#############" + str(idx) + "/" + str(len(test_instances)) + "################")
            print ("################################")


            results.append(return_list)

        
    else: 
        for idx in range(num_runs):
            

            return_list, inst = test_graph_instance(subsets = None,
                                                universe = None,
                                                solutions = None,
                                                cost_type = cost_type,
                                                costs = None,
                                                model_path = model_path,
                                                feats = feats,
                                                dec_size = dec_size, 
                                                perc_start = perc_start,
                                                init_sol_perc = init_sol_perc,
                                                obj_threshold = obj_threshold, 
                                                m_type = m_type, 
                                                time_limit = time_limit, 
                                                run_random = run_random, 
                                                gnn_solver = gnn_solver,
                                                neurons = neurons)
            print ("################################")
            print ("#############" + str(idx) + "################")
            print ("################################")
            
            instances.append(inst)

            results.append(return_list)
        
    return results, instances
    
    
def scip_solve(universe, subsets, costs, init_sets = None):
    
    s = time.time()
    # Create Model
    model = pyscipopt.Model("Set Cover")

    # create a binary variable for each subset
    vars = []
    for i in range(len(subsets)):
        var = model.addVar("x_{}".format(i), vtype="B")
        vars.append(var)

    # set the objective function to minimize the total cost of the selected subsets
    model.setObjective(sum(costs[i] * vars[i] for i in range(len(subsets))), "minimize")

    # add a constraint for each element in the ground set
    # the element must be covered by at least one of the selected subsets
    for e in universe:
        model.addCons(sum(vars[i] for i in range(len(subsets)) if e in subsets[i]) >= 1)

    # solve the model
    model.setParam("presolving/maxrestarts", 0)
    model.setParam("separating/maxrounds", 0)
    model.hideOutput(True)
    
    if init_sets is not None: 
        m_vars = model.getVars()
        for idx in init_sets.keys():
            model.chgVarBranchPriority(m_vars[idx], 2)
    
    model.optimize()
    
    best_sol = model.getBestSol()
    variables = model.getVars()
    sol = []
    for idx, var in enumerate(variables):
        if model.getVal(var) == 1:
            sol.append(str(idx))


    
    return [sol], time.time() - s, model.getNNodes(), model.getObjVal()

class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores (expensive expert)
    or pseudocost scores (weak expert for exploration) when called at every node.
    """

    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        """
        This function will be called at initialization of the environment (before dynamics are reset).
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        """
        probabilities = [1 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)
        
        
class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        candidates,
        nb_candidates,
        candidate_choice,
        candidate_scores,
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], "rb") as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample
        
        constraint_features = sample_observation.row_features
        edge_indices = sample_observation.edge_features.indices.astype(np.int32)
        edge_features = np.expand_dims(sample_observation.edge_features.values, axis=-1)
        variable_features = sample_observation.variable_features

        # We note on which variables we were allowed to branch, the scores as well as the choice
        # taken by strong branching (relative to the candidates)
        candidates = np.array(sample_action_set, dtype=np.int32)
        candidate_scores = np.array([sample_scores[j] for j in candidates])
        candidate_choice = np.where(candidates == sample_action)[0][0]

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
            torch.LongTensor(candidates),
            len(candidates),
            torch.LongTensor([candidate_choice]),
            torch.FloatTensor(candidate_scores)
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]

        return graph
    
class GNNPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)
        return output


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__("add")
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output
    
def process(policy, data_loader, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(DEVICE)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            # Index the results by the candidates, and split and pad them
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            # Compute the usual cross-entropy classification loss
            loss = F.cross_entropy(logits, batch.candidate_choices)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
            accuracy = (
                (true_scores.gather(-1, predicted_bestindex) == true_bestscore)
                .float()
                .mean()
                .item()
            )

            mean_loss += loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_acc /= n_samples_processed
    return mean_loss, mean_acc


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack(
        [
            F.pad(slice_, (0, max_pad_size - slice_.size(0)), "constant", pad_value)
            for slice_ in output
        ],
        dim=0,
    )
    return output

        
# We can pass custom SCIP parameters easily
scip_parameters = {
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "limits/time": 3600,
}

# Note how we can tuple observation functions to return complex state information
env = ecole.environment.Branching(
    observation_function=(
        ExploreThenStrongBranch(expert_probability=0.05),
        ecole.observation.NodeBipartite(),
    ),
    scip_params=scip_parameters,
)

# This will seed the environment for reproducibility
env.seed(0)

scip_parameters = {
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "limits/time": 3600,
}
env = ecole.environment.Branching(
    observation_function=ecole.observation.NodeBipartite(),
    information_function={
        "nb_nodes": ecole.reward.NNodes(),
        "time": ecole.reward.SolvingTime(),
    },
    scip_params=scip_parameters,
)

env_gg = ecole.environment.Branching(
    observation_function=ecole.observation.NodeBipartite(),
    information_function={
        "nb_nodes": ecole.reward.NNodes(),
        "time": ecole.reward.SolvingTime(),
    },
    scip_params=scip_parameters,
)

default_env = ecole.environment.Configuring(
    observation_function=None,
    information_function={
        "nb_nodes": ecole.reward.NNodes(),
        "time": ecole.reward.SolvingTime(),
    },
    scip_params=scip_parameters,
)

gscp_env = ecole.environment.Configuring(
    observation_function=None,
    information_function={
        "nb_nodes": ecole.reward.NNodes(),
        "time": ecole.reward.SolvingTime(),
    },
    scip_params=scip_parameters,
)

def train_graph_instance_unsup(subsets, universe, solutions, m_type = 'sage', cost_type = 'length', costs = None, epochs = 500, patience = 250, model_path = None, model_name = None, feats = None):

    return_dict = create_graph(universe = universe,
                               subsets = subsets)
    
    if cost_type == 'equal':
        solutions, _, _ = gurobi_solver(subsets = subsets,
                             universe = universe,
                             cost_type = cost_type,
                             costs = None,
                             initial_values = None,
                             time_limit = None,
                             stop_obj = None,
                             callback = False,
                             return_model_only = False)
        optimal_objective = np.sum([1 for s in solutions[0]])
        
    elif cost_type == 'length':
        optimal_objective = np.sum([len(subsets[int(s)]) for s in solutions[0]])
    else: 
        optimal_objective = np.sum([costs[int(s)] for s in solutions[0]])
        og_costs = costs.copy()
    
    gu_model = gurobi_solver(subsets = subsets,
                             universe = universe,
                             cost_type = cost_type,
                             costs = costs,
                             initial_values = None,
                             time_limit = None,
                             stop_obj = None,
                             callback = False,
                             return_model_only = True)
    constraint_mat = torch.tensor(gu_model.getA().todense(), dtype = torch.float32).to(device)
    cost_mat = np.array(gu_model.getAttr('Obj'))
    if costs is not None:
        cost_mat = (cost_mat - np.min(cost_mat)) / np.ptp(cost_mat)
    cost_mat = torch.tensor(cost_mat, dtype = torch.float32).to(device).reshape(-1, 1)

    graph = return_dict['graph'] 
    hyp_graph = return_dict['hyp_graph'] 
    uni_graph = return_dict['uni_graph'] 
    set_to_idx = return_dict['set_to_idx'] 
    idx_to_set = return_dict['idx_to_set']
    hyp_sti = return_dict['hyp_sti']
    hyp_its = return_dict['hyp_its'] 
    set_nodes_mapped = return_dict['set_nodes_mapped'] 
    element_nodes_mapped = return_dict['element_nodes_mapped'] 
    hyp_edge_weight = return_dict['hyp_edge_weight'] 
    hyp_node_weight = return_dict['hyp_node_weight']
    set_nodes = return_dict['set_nodes']
    element_nodes = return_dict['element_nodes']
    
    set_nodes = []
    for key in set_to_idx:
        k = eval(key)
        if type(k) == type(set()):
            if k != universe:
                set_nodes.append(set_to_idx[key])
                
    element_nodes = []
    for key in set_to_idx:
        k = eval(key)
        if type(k) == type(5):

            element_nodes.append(set_to_idx[key])
    

    if costs is not None:
        subset_to_cost = {str(sorted(v)) : costs[k] for k, v in subsets.items()}
        remapped_costs = {}
        for idx in range(len(graph)):
                
            if str(idx_to_set[idx]) == str(universe):
                remapped_costs[idx] = 0
            
            elif '{' not in str(idx_to_set[idx]):
                remapped_costs[idx] = 0
                
            else: 
                remapped_costs[idx] = subset_to_cost[str(sorted(eval(idx_to_set[idx])))]
                
        
        costs = remapped_costs
         
    
    if cost_type == 'length':
        costs_vec = np.array([len(idx_to_set[node]) for node in graph]).reshape(-1, 1)
    elif cost_type == 'equal':
        costs_vec = np.array([1 for node in graph]).reshape(-1, 1)
    elif cost_type == 'custom':
        costs_vec = np.array(list(costs.values())).reshape(-1, 1)
    
    if cost_type != 'equal':
        costs_vec = (costs_vec - np.min(costs_vec)) / np.ptp(costs_vec)
    
    cat_vec = np.zeros((len(graph), 1))
    universe_node = set_to_idx[str(universe)]
    element_nodes = []
    for node_set, node_id in set_to_idx.items():
        if type(eval(node_set)) == int:
            element_nodes.append(node_id)
            cat_vec[node_id] = 1
    cat_vec[universe_node] = 1
    
    
    # Hypergraph Features
    graph_sf, uni_vec, el_pr_vec, sub_pr_vec, cover_vec = get_hyp_features(return_dict, universe)

    feat_list = []
    if 'bi_sf' in feats:
        feat_list.append(graph_sf)
        
    if 'uni_sf' in feats: 
        feat_list.append(uni_vec)
        
    if 'hyp_feat' in feats:
        feat_list.append(el_pr_vec)
        feat_list.append(sub_pr_vec)
    
    if 'ppr' in feats:
        vec = {idx : 0 for idx in range(len(graph))}
        vec[set_to_idx[str(universe)]] = 1
        ppr_dict = nx.pagerank(graph, personalization = vec)
        ppr_vec = list(dict(sorted(nx.pagerank(graph, personalization = vec).items())).values())
        ppr_vec = np.array(ppr_vec).reshape(-1, 1)
        ppr_vec = (ppr_vec - np.min(ppr_vec, axis = 0)) / np.ptp(ppr_vec, axis = 0)
        feat_list.append(ppr_vec)
        
    if 'costs' in feats:
        feat_list.append(costs_vec)
        
    if 'cat' in feats:
        feat_list.append(cat_vec)
        
    if 'cover' in feats: 
        cover_vec = (cover_vec - np.min(cover_vec, axis = 0)) / np.ptp(cover_vec, axis = 0)
        feat_list.append(cover_vec)
        
    features = np.hstack(feat_list)
    print (features.shape)
        
    edge_index = np.array(list(nx.to_edgelist(graph)))[:, :2].T.astype(int)

    data = Data(x = torch.tensor(features.astype(np.float32)),\
                edge_index = torch.tensor(edge_index))

    # data = T.NormalizeFeatures()(data)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    
    y = np.zeros((len(graph), 1))
    
    # Solution sets
    # Build subsets from indices
    for sol in solutions: 
        current_sol = [str(subsets[int(s)]) for s in sol]
        
    # Map subsets to nodes
    for sol in current_sol:
        node = set_to_idx[str(sol)]
        y[node] = 1
        
    # Element nodes  
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Class labels
    _, counts = np.unique(y, return_counts = True)
    class_weights = 1 - (counts / np.sum(counts))
    weight = [class_weights[int(y[idx])] for idx in range(len(y))]
    print ("Class Counts: ", class_weights)
    print ("Graph Size: ", len(graph))
    y = torch.tensor(y)
    y = y.type(torch.float32)
    y = y.to(device)
    weight = torch.tensor(weight).reshape(-1, 1)# + 1
    weight = weight.type(torch.float32)
    weight = weight.to(device)
    
    # Create model
    if m_type == 'gat':
        model = GAT(hidden = 32, 
                      in_head = 16, 
                      out_head = 64, 
                      num_features = data.x.shape[1], 
                      out_dim = 64)
        
    elif m_type == 'sage':
        model = SAGE(in_channels = data.x.shape[1],
                 hidden_channels = 1024,
                 out_channels = 1024,
                 num_layers = 2,
                 dropout = 0.4).to(device)
        
    elif m_type == 'gin':
        model = GIN(in_channels = data.x.shape[1],
                 hidden_channels = 1024,
                 out_channels = 1024,
                 num_layers = 2,
                 dropout = 0.4).to(device)
        
    elif m_type == 'gcn':
        model = GCN(in_channels = data.x.shape[1],
                 hidden_channels = 1024,
                 out_channels = 1024,
                 num_layers = 2,
                 dropout = 0.4).to(device)
        
    elif m_type == 'gin':
        model = gin(in_channels = data.x.shape[1],
                 hidden_channels = 1024,
                 out_channels = 1024,
                 num_layers = 2,
                 dropout = 0.4).to(device)
        
    elif m_type == 'cheb':
        model = Cheb(in_channels = data.x.shape[1],
                 hidden_channels = 1024,
                 out_channels = 1024,
                 num_layers = 2,
                 dropout = 0.4).to(device)

    model.to(device)
    model.optimizer.zero_grad()

    if model_path is not None:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else: 
            model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
           
    model.optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 5e-4)
    model.train()

    # Train
    best = 1e9
    wait = 0
    size_skip = False
    
    for ep in range(epochs):
        
        model.optimizer.zero_grad()
        out = model(x, edge_index)
        sub_preds = out[set_nodes].reshape(-1, 1)
        y_sub = y[set_nodes].reshape(-1, 1)
        sup_loss = F.binary_cross_entropy(sub_preds, y_sub)
        # sup_loss = F.binary_cross_entropy(out, y)
        
        # Extract the right nodes 
        sol_mat = out[set_nodes].reshape(-1, 1)
        
        try:
            constaint_sol = constraint_mat @ sol_mat
            sol_mat = sol_mat * cost_mat
            unsup_loss = torch.square(torch.sum(sol_mat)) - (0.4 * torch.sum(1 - constaint_sol)) - torch.sum(constaint_sol - 1)
            # unsup_loss = torch.sum(sol_mat)

            # print ("Sup", sup_loss.item())
            # print ("Unsup", 1e-4 * unsup_loss.item())
           
            loss = (1.0 * sup_loss) + (1e-4 * unsup_loss)
            
        except: 
            print ("Skipping Unsup")
            size_skip = True
            loss = sup_loss
            
        print ('Epoch: ', str(ep), '(', loss.item() ,')', end = '\r')
        
        if loss < best:
            best = loss
            wait = 0

            torch.save(model.state_dict(), '/home/shafi.z/Graph-SCP/' + model_name + '.pkl')
        else:
            wait = wait + 1

        if wait == patience: 
            print ("Early Stopping")
            break

        loss.backward()
        model.optimizer.step()
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('/home/shafi.z/Graph-SCP/' + model_name + '.pkl'))
    else: 
        model.load_state_dict(torch.load('/home/shafi.z/Graph-SCP/' + model_name + '.pkl', map_location = torch.device('cpu')))
        
    
    # Evaluation
    model.eval()
    out = model(x, edge_index)
    np_out = out.cpu().detach().numpy().reshape(-1, )  
    y = y.cpu().detach().numpy()

    # Find Threshold 
    thresholds = np.arange(0, 1.02, 0.02)
    f1_t_list = []
    for t in thresholds: 
        f1_t_list.append(f1_score(y, (np_out > t).astype(int)))

    f1 = max(f1_t_list)
    threshold = thresholds[np.argmax(f1_t_list)]

    auc = roc_auc_score(y, np_out)
    aup = average_precision_score(y, np_out)
    acc = accuracy_score(y, (np_out > threshold).astype(int))
    conf = confusion_matrix(y, (np_out > threshold).astype(int))

    print ("F1 : ", f1)
    print ("AUC : ", auc)
    print ("AUP :", aup)
    print ("Acc : ", acc)
    
    del model

    print ("Output:")
    # plt.hist(np_out, bins = 50)
    # plt.show()

    return f1, auc, aup, acc, np.divide(*class_weights), size_skip

def train_set_cover_unsup(train_instances, cost_type, epochs, patience, model_path, model_name, feats, m_type):
    
    f1_list = []
    auc_list = []
    aup_list = []
    acc_list = []
    ratio_list = []
    size_skip_count = 0
    
    for idx in range(len(train_instances)):
        print ("Instance : ", idx, "/", len(train_instances))
        if train_instances[idx]['solutions'] != []:
            
            if cost_type == 'custom':
                costs = train_instances[idx]['costs']
            else: 
                costs = None
            

            f1, auc, aup, acc, ratio, size_skip = train_graph_instance_unsup(subsets = train_instances[idx]['subsets'],
                                                     universe = train_instances[idx]['universe'],
                                                     solutions = train_instances[idx]['solutions'],
                                                     cost_type = cost_type,
                                                     costs = costs,
                                                     epochs = epochs,
                                                     patience = patience,
                                                     model_path = model_path,
                                                     model_name = model_name, 
                                                     feats = feats, 
                                                     m_type = m_type)

            f1_list.append(f1)
            auc_list.append(auc)
            aup_list.append(aup)
            acc_list.append(acc)
            ratio_list.append(ratio)
            if size_skip: 
                size_skip_count += 1

            if model_path is None:
                model_path = '/home/shafi.z/Graph-SCP/' + model_name + '.pkl'

    print (size_skip_count, "/", len(train_instances)) 
    return f1_list, auc_list, aup_list, auc_list, ratio_list