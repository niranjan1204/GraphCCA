import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import math, json, argparse, sys
import pickle as pkl
import networkx as nx
import gensim

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K


# Some Functions

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))

    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    test_edge_idx = all_edge_idx[:num_test]
    test_edges = edges[test_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(train_edges_false, edges_all)
    assert ~ismember(test_edges, train_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # These edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, test_edges, test_edges_false


def deepwalk(adj, num_vertices):
    adj_list = [list(line.nonzero()[1]) for line in adj]
    sequence = []
    context = []
    # Perform random walks
    for i in range(max_iter):
        for v in range(num_vertices):
            walk = [v]
            prev_nbr = v
            for i in range(1, walk_len):
                next_nbr = int(np.random.uniform(low=0, high=len(adj_list[prev_nbr])))
                if len(adj_list[prev_nbr]) == 0:
                    break
                walk.append(adj_list[prev_nbr][next_nbr])
                prev_nbr = next_nbr
            for i in range(len(walk)):
                walk[i] = str(walk[i])
            sequence.append(walk)
    # Implement CBoW on the squences
    model = gensim.models.Word2Vec(sequence, size = node_size)
    model.train(sequence, total_examples=len(sequence), epochs=max_iter)
    for i in range(num_vertices):
        context.append(np.array(model.wv[str(i)]))
    return np.array(context)


def data_embed(Y):
    # fix random seed for reproducibility
    np.random.seed(7)
    
    l, m = Y.shape
    n = int(math.sqrt(m*node_size))

    # create model
    model = Sequential()
    model.add(Dense(n, kernel_initializer ='normal', activation = 'relu', input_dim = m))
    model.add(Dropout(0.1))
    model.add(Dense(node_size, kernel_initializer ='normal', activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(n, kernel_initializer ='normal', activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(m, kernel_initializer ='normal', activation = 'relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model
    model.fit(Y, Y, epochs = 200, batch_size = 100,  verbose = 0)

    # Return the Embedding
    get_layer_output = K.function([model.layers[0].input], [model.layers[2].output])
    Z = get_layer_output([Y])[0]
    return Z


def CCAnalysis(X, S):
    # Initialization of variables
    N = X.shape[0]
    D_x, D_s, D_z = X.shape[1], S.shape[1], embed_size
    X, S = X.T, S.T
    alpha0, beta0 = 1e-5, 1e-5
    tau1_mean, tau2_mean = 1e-1, 1e-1
    
    X = X*math.sqrt(N*D_x)/np.linalg.norm(X)
    S = S*math.sqrt(N*D_s)/np.linalg.norm(S)

    X_frob = np.linalg.norm(X)
    S_frob = np.linalg.norm(S)

    alpha1_mean = np.ones(D_z)/math.sqrt(D_z)
    alpha2_mean = np.ones(D_z)/math.sqrt(D_z)

    A1_cov = np.identity(D_z)
    A1_mean = np.ones((D_x, D_z))/math.sqrt(D_z)

    A2_cov=np.identity(D_z)
    A2_mean = np.ones((D_s, D_z))/math.sqrt(D_z)

    z_cov=np.identity(D_z)
    z_mean=np.ones((D_z, N))/math.sqrt(D_z)

    # BCCA Updates till convergence
    while True:
        zzT = np.dot(z_mean, z_mean.T) + np.dot(D_z, z_cov)

        A1_cov = np.linalg.inv(np.diag(alpha1_mean) + (tau1_mean)*zzT)
        A1_mean = np.dot(np.dot(X, z_mean.T), (tau1_mean)*A1_cov)
        A1TA1_mean = np.dot(A1_mean.T, A1_mean) + np.dot(D_x, A1_cov)

        A2_cov = np.linalg.inv(np.diag(alpha2_mean) + (tau2_mean)*zzT)
        A2_mean = np.dot(np.dot(S, z_mean.T), (tau2_mean)*A2_cov)
        A2TA2_mean = np.dot(A2_mean.T, A2_mean) + np.dot(D_s, A2_cov)
        
        z_old = z_mean
        z_cov = np.linalg.inv(np.identity(D_z) + (tau1_mean)*A1TA1_mean + (tau2_mean)*A2TA2_mean)
        z_mean = np.dot((tau1_mean)*z_cov, np.dot(A1_mean.T, X)) + np.dot((tau2_mean)*z_cov, np.dot(A2_mean.T, S))

        for k in range(D_z):
            alpha1_mean[k]=(beta0+A1TA1_mean[k][k]/2.0)/(alpha0 + D_x/2.0)
            alpha2_mean[k]=(beta0+A2TA2_mean[k][k]/2.0)/(alpha0 + D_s/2.0)

        tau1_mean = (alpha0+N*D_x/2.0)/(beta0+(X_frob**2 + np.trace(np.dot(A1TA1_mean, zzT))-2*np.trace(np.dot(np.dot(A1_mean, z_mean), X.T)))/2.0) 
        tau2_mean = (alpha0+N*D_s/2.0)/(beta0+(S_frob**2 + np.trace(np.dot(A2TA2_mean, zzT))-2*np.trace(np.dot(np.dot(A2_mean, z_mean), S.T)))/2.0)

        if  np.linalg.norm(z_old-z_mean) < (epsilon*np.linalg.norm(z_mean)):
            break
        else:
            z_old = z_mean

    return z_mean.T


def main():

    '''	DATASETS:
		cora == (2708, 1433) , edges: 7333264
		citeseer == (3327, 3703) , edges: 11068929
		pubmed == (19717, 500)
    '''

    print('READING DATA')
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('-config', help='Config to read details', required=True)
    args = parser.parse_args()

    with open(args.config) as configfile:
        config = json.load(configfile)
    
    global dataset_str, train_edges, train_edges_false, test_edges, test_edges_false, z_matrix, epsilon, node_size, embed_size, max_iter, walk_len
    
    dataset_str = config["dataset"]
    train_edges_path = config["train_edges"]
    train_edges_false_path = config["train_edges_false"]
    test_edges_path = config["test_edges"]
    test_edges_false_path = config["test_edges_false"]
    Z_matrix_path = config["z_matrix"]
    A_matrix_path = config["a_matrix"]
    C_matrix_path = config["c_matrix"]
    epsilon = config["epsilon"]
    embed_size = config["embed_size"]
    node_size = config["node_size"]
    max_iter = config["max_iter"]
    walk_len = config["walk_len"]

    # Load data
    adj, features = load_data(dataset_str)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

    # Normalizing the feature embeddings
    print('FEATURE COMPRESSION')
    features_raw = features.toarray()
    features_raw = features_raw*(math.sqrt(features.shape[0]*features.shape[1]))/np.linalg.norm(features_raw)
    features = data_embed(features_raw)

    # Deepwalk
    print('BUILDING NODE VECTORS')
    nodevec_embed = deepwalk(adj_train, adj_train.shape[0])

    # Simple Concatenation
    concat_embed = np.concatenate((features, nodevec_embed), axis=1)

    # Graph CCA
    print('PERFORMING CCA')
    Z_matrix = CCAnalysis(features, nodevec_embed)
    
    # Save the data
    np.save(train_edges_path, train_edges)
    np.save(train_edges_false_path, train_edges_false)
    np.save(test_edges_path, test_edges)
    np.save(test_edges_false_path, test_edges_false)
    np.save(C_matrix_path, concat_embed)
    np.save(A_matrix_path, nodevec_embed)
    np.save(Z_matrix_path, Z_matrix)
    

if __name__ == '__main__':
    main()
