import numpy as np
import networkx as nx
import sparse
import nltk
import cPickle as pickle
import os
from setting import *
from joblib import Parallel, delayed


def get_proof_graph(ppr_grounded_line, feature_vector_size):

    graph = nx.DiGraph()
    array = ppr_grounded_line.strip().split('	')
    query_example = array[0]
    query_node = int(array[1])

    pos_nodes = map(int, array[2].split(','))
    neg_nodes = map(int, array[3].split(','))

    nodes_count = int(array[4])
    edges_count = int(array[5])

    label_dependencies_count = int(array[6])

    edges = array[7:]

    nodes = []
    for e in ppr_grounded_line.strip().split('	'):
        if '->' in e:
            nodes.append(int(e.split('->')[0]))
            nodes.append(int(e.split('->')[1].split(':')[0]))
    nodes = list(set(nodes))

    for node in nodes:
        if node in pos_nodes:
            graph.add_node(node, Label=1)
        elif node in neg_nodes:
            graph.add_node(node, Label=-1)
        elif node == query_node:
            graph.add_node(node, Label=2)
        else:
            graph.add_node(node, Label=0)

    for edge in edges:
        source, target, feature_weights = edge.replace('->', ':').split(':')
        source = int(source)
        target = int(target)
        feature_weights = [
            feature.split('@') for feature in feature_weights.split(',')
        ]
        feature_weights = [(int(feature_weight[0]), float(feature_weight[1]))
                           for feature_weight in feature_weights]

        vector = [0.0] * feature_vector_size
        for feature_weight in feature_weights:
            vector[feature_weight[0]] = feature_weight[1]
        # graph.add_edge( source, target, {'feature_vector': ",".join( map(str, vector) ) } )
        graph.add_edge(source, target, {'feature_vector': vector})

    # nx.write_graphml( graph, "graph.graphml" )
    return graph


def get_proof_graph_tensor(proof_graph, feature_vector_size):

    node_list = proof_graph.nodes()

    adjacency_matrix = nx.adjacency_matrix(proof_graph, weight=None)
    adjacency_matrix = adjacency_matrix.astype(float)

    size = len(node_list)
    featured_adjacency_matrix = np.array([[[0.0] * feature_vector_size
                                           for x in range(size)]
                                          for y in range(size)])

    for edge in proof_graph.edges_iter():
        source = edge[0]
        target = edge[1]
        source_index = node_list.index(edge[0])
        target_index = node_list.index(edge[1])
        feature_vector = proof_graph[source][target]['feature_vector']
        featured_adjacency_matrix[source_index][target_index] = feature_vector

    featured_adjacency_matrix = np.reshape(featured_adjacency_matrix,
                                           [size, size, -1])

    correct_answer_vector = np.zeros([size, 1], dtype=np.float32)
    incorrect_answer_vector = np.zeros([size, 1], dtype=np.float32)
    one_hot_query_vector = np.zeros([size, 1], dtype=np.float32)

    for node_index, node in enumerate(proof_graph.nodes()):
        if int(proof_graph.node[node]["Label"]) == 1:
            correct_answer_vector[node_index] = 1.0
        else:
            correct_answer_vector[node_index] = 0.0

    for node_index, node in enumerate(proof_graph.nodes()):
        if int(proof_graph.node[node]["Label"]) == -1:
            incorrect_answer_vector[node_index] = 1.0
        else:
            incorrect_answer_vector[node_index] = 0.0

    for node_index, node in enumerate(proof_graph.nodes()):
        if int(proof_graph.node[node]["Label"]) == 2:
            one_hot_query_vector[node_index] = 1.0
        else:
            one_hot_query_vector[node_index] = 0.0

    return [
        one_hot_query_vector, featured_adjacency_matrix, correct_answer_vector,
        incorrect_answer_vector
    ]


##### (END) Conversion of question objects/graphs to feature representations #####


def dump_graph_tensor(idx, tensors_dir, ppr_grounded_line):

    feature_vector_size = int(ppr_grounded_line.strip().split('\t')[6]) + 1
    proof_graph = get_proof_graph(ppr_grounded_line, feature_vector_size)
    data = get_proof_graph_tensor(proof_graph, feature_vector_size)
    sparse_data = [sparse.COO(item) for item in data]

    sparse_tensor_path = os.path.join(tensors_dir,
                                      '{}-sparse.pickle'.format(idx))
    with open(sparse_tensor_path, 'wb') as g:
        pickle.dump(sparse_data, g, protocol=pickle.HIGHEST_PROTOCOL)

    return feature_vector_size


# ppr_grounded_line = 'predict(train00004,X1).	1	6	5	6	13	42	6->6:9@1.0	6->1:2@1.0	5->5:9@1.0	5->1:2@1.0	4->6:10@0.6097,14@0.4334,13@0.6097,12@0.2572,11@0.051	4->1:2@1.0	3->5:8@0.6097,7@0.4334,6@0.2572,5@0.6097,4@0.051	3->1:2@1.0	2->3:3@1.0	2->4:3@1.0	2->1:2@1.0	1->2:1@1.0	1->1:2@1.0'

processed_data_dir = os.path.join('ProcessedData', program_name)
set_names = ['train', 'test']

process_count = 4
for set_name in set_names:
    print 'In set {}'.format(set_name)
    sld_grounded_path = os.path.join(
        processed_data_dir, program_name + '-{}.grounded'.format(set_name))
    tensors_dir = os.path.join(processed_data_dir, 'Tensors', set_name)

    if not os.path.exists(tensors_dir):
        os.makedirs(tensors_dir)

    with open(sld_grounded_path) as f:
        ppr_grounded_lines = f.readlines()
        feature_vector_sizes = Parallel(n_jobs=process_count)(
            delayed(dump_graph_tensor)(idx, tensors_dir, ppr_grounded_line)
            for idx, ppr_grounded_line in enumerate(ppr_grounded_lines))
        feature_vector_size = feature_vector_sizes[0]

    print 'set {} processed'.format(set_name)
    feature_vector_size_path = os.path.join(processed_data_dir, 'feat_size.txt')
    with open(feature_vector_size_path, 'w') as f:
        f.write(str(feature_vector_size))
