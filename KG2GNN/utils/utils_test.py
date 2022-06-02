import numpy as np
import scipy.sparse as sp
import torch
import time
import random

from utils.tool import read_data, write_dic, dictionary, normalize, sparse_mx_to_torch_sparse_tensor


def encoding_test(run=10, train_dataset="fb237_v1", test_dataset="fb237_v1_ind"):
    """load test-graph and test-facts, and do the encoding on the test-graph"""

    print("Start to encoding test-graph and load test-facts for {}".format(test_dataset))

    t_start = time.time()

    path = "data"

    test_graph_path = "{}/{}/test/test-graph.txt".format(path, test_dataset)
    test_fact_path = "{}/{}/test/test-random-sample/test{}.txt".format(path, test_dataset, run)
    train_path = "{}/{}/train/train.txt".format(path, test_dataset)
    valid_path = "{}/{}/train/valid.txt".format(path, test_dataset)
    train_triples = read_data(train_path)
    valid_triples = read_data(valid_path)
    triples_for_filter = set()
    for t in train_triples:
        triples_for_filter.add((t[0], t[1], t[2]))
    for t in valid_triples:
        triples_for_filter.add((t[0], t[1], t[2]))
    # these two paths are for loading
    relation_dic_path = "{}/{}/train/relation-dic.txt".format(path, train_dataset)
    type_dic_path = "{}/{}/train/type-dic.txt".format(path, train_dataset)

    # these two paths are for generating
    constant_dic_path = "{}/{}/test/test-random-sample/test-constant-dic{}.txt".format(path, test_dataset, run)
    pair_dic_path = "{}/{}/test/test-random-sample/test-pair-dic{}.txt".format(path, test_dataset, run)

    test_graph_triples = read_data(test_graph_path)
    test_fact_triples_with_label = read_data(test_fact_path)

    # load relation dic and type dic generated by training
    f_relation_dic = open(relation_dic_path)
    relations = []
    for line in f_relation_dic:
        relation_new = line.strip().split("\t")[1]
        relations.append(relation_new)
    relation_set = set(relations)

    f_type_dic = open(type_dic_path)
    types = []
    for line in f_type_dic:
        type_new = line.strip().split("\t")[1]
        types.append(type_new)

    all_triples_with_label = test_graph_triples + test_fact_triples_with_label
    for t in all_triples_with_label:
        if t[-1] != "0":
            triples_for_filter.add((t[0], t[1], t[2]))

    test_graph_real_triples = []
    test_graph_type_triples = []
    for triple in test_graph_triples:
        if triple[1] != "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
            test_graph_real_triples.append(triple)
        else:
            test_graph_type_triples.append(triple)

    test_fact_real_triples_with_label = []
    test_fact_type_triples_with_label = []
    for triple in test_fact_triples_with_label:
        if triple[1] != "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
            test_fact_real_triples_with_label.append(triple)
        else:
            test_fact_type_triples_with_label.append(triple)

    all_real_triples_with_label = []
    all_type_triples_with_label = []

    constant_set = set()
    for triple in all_triples_with_label:
        if triple[1] != "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
            constant_set.add(triple[0])
            constant_set.add(triple[2])
            all_real_triples_with_label.append(triple)
        else:
            constant_set.add(triple[0])
            all_type_triples_with_label.append(triple)

    constants = list(constant_set)

    constant2index = dictionary(constants)
    relation2index = dictionary(relations)
    type2index = dictionary(types)

    # generate list of pairs for encoding

    pairs = []
    pair_set = set()
    for triple in all_real_triples_with_label:
        sub_idx = constant2index[triple[0]]
        obj_idx = constant2index[triple[2]]
        if sub_idx < obj_idx:
            if (sub_idx, obj_idx) not in pair_set:
                pair_set.add((sub_idx, obj_idx))
                pairs.append((sub_idx, obj_idx))
        if sub_idx > obj_idx:
            if (obj_idx, sub_idx) not in pair_set:
                pair_set.add((obj_idx, sub_idx))
                pairs.append((obj_idx, sub_idx))

    edge = torch.Tensor(pairs).t()

    r_hits_candidates_triple = []

    for triple in test_fact_triples_with_label:
        if triple[-1] == "1":
            r_hits_candidates_for_t = []
            for candidate in relations:
                if candidate != triple[1]:
                    if (triple[0], candidate, triple[2]) not in triples_for_filter:
                        r_hits_candidates_for_t.append([triple[0], candidate, triple[2]])
            r_hits_candidates_triple.append(r_hits_candidates_for_t)

    #for constant_idx in range(len(constants)):
    #    pairs.append((constant_idx, constant_idx))
    #    pair_set.add((constant_idx, constant_idx))
    pair2index = dictionary(pairs)

    write_dic(constant_dic_path, constants)
    write_dic(pair_dic_path, pairs)

    def triple2index(triple_now):
        sub_idx = constant2index[triple_now[0]]
        relation_idx = relation2index[triple_now[1]]
        obj_idx = constant2index[triple_now[2]]
        if (sub_idx, obj_idx) in pair_set:
            pair_idx = pair2index[(sub_idx, obj_idx)]
            dim_idx = len(types) + relation_idx
        elif (obj_idx, sub_idx) in pair_set:
            pair_idx = pair2index[(obj_idx, sub_idx)]
            dim_idx = len(types) + len(relations) + relation_idx
        else:
            print(triple_now, sub_idx, relation_idx, obj_idx)
            print("wrong")
        return pair_idx, dim_idx

    r_hits_candidates = []

    hits_true = []
    triple_idx = 0

    for triple in test_fact_triples_with_label:
        if triple[-1] == "1":

            r_hits_candidates_for_t_ori = r_hits_candidates_triple[triple_idx]
            r_hits_candidates_for_t = []

            for t in r_hits_candidates_for_t_ori:
                candidate = triple2index(t)
                r_hits_candidates_for_t.append(candidate)
            r_hits_candidates.append(r_hits_candidates_for_t)
            hits_true.append(triple2index(triple))
            triple_idx += 1

    print(
        "Num of triples in test-graph:{}, Num of triples for testing:{}, num of pairs:{}, num of constants:{}, num of relations:{}, num of types:{}".format(
            len(test_graph_triples), len(test_fact_triples_with_label), len(pairs), len(constants), len(relations),
            len(types)))

    print("Start to encode the graph")
    s_time = time.time()

    # collect related pairs for each constant

    pairs_for_constant = dict([(i, set()) for i in range(len(constants))])
    p_idx = 0
    for pair in pairs:
        p_idx = pair2index[pair]
        c1 = pair[0]
        c2 = pair[1]
        pairs_for_constant[c1].add(p_idx)
        pairs_for_constant[c2].add(p_idx)

    # collect neighbors for each pair node

    pneighbors_for_pair = dict([(i, set()) for i in range(len(pairs))])
    for c_idx in range(len(constants)):
        pairs_c = set(pairs_for_constant[c_idx])
        # pair and n_pair would contain one common constant
        for pair in pairs_c:
            for n_pair in pairs_c:
                if pair != n_pair:
                    pneighbors_for_pair[pair].add(n_pair)

                    # generate edge list

    edges = []

    for i in range(len(pairs)):
        pneighbors = pneighbors_for_pair[i]
        for pneighbor in pneighbors:
            edges.append([i, pneighbor])
            edges.append([pneighbor, i])

    print("Finished generating edges", time.time() - s_time)

    # generate a normalized adjencency matrix (strategy for GCN)

    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(pairs), len(pairs)),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # del edges
    print("Total time for adj: {:.4f}s".format(time.time() - s_time))

    print("Start to generate features, labels, and masks")

    def initialize(test_graph_real_triples, test_graph_type_triples, test_fact_real_triples_with_label,
                   test_fact_type_triples_with_label):

        labels = torch.zeros(len(pairs), len(types) + 2 * len(relations))
        masks = torch.zeros(len(pairs), len(types) + 2 * len(relations))
        features = torch.zeros(len(pairs), len(types) + 2 * len(relations))

        # labels and masks are generated for all triples in test-facts (pos&neg)

        for triple in test_fact_type_triples_with_label:
            cons = triple[0]
            typ = triple[2]
            label = triple[3]
            pair_idx = pair2index[(constant2index[cons], constant2index[cons])]
            typ_idx = type2index[typ]
            if label == "1":
                labels[pair_idx][typ_idx] = 1
            elif label == "0":
                labels[pair_idx][typ_idx] = 0
            masks[pair_idx][typ_idx] = 1

        for triple in test_fact_real_triples_with_label:
            sub = triple[0]
            rel = triple[1]
            obj = triple[2]
            label = triple[3]
            sub_idx = constant2index[sub]
            rel_idx = relation2index[rel]
            obj_idx = constant2index[obj]

            if sub_idx == obj_idx:
                continue

            try:
                pair_idx = pair2index[(sub_idx, obj_idx)]
            except:
                pair_idx = pair2index[(obj_idx, sub_idx)]
                rel_idx = rel_idx + len(relations)
            if label == "1":
                labels[pair_idx][len(types) + rel_idx] = 1
            elif label == "0":
                labels[pair_idx][len(types) + rel_idx] = 0
            masks[pair_idx][len(types) + rel_idx] = 1

            # features are generated for all triples in test-graph (pos&neg)

        for triple in test_graph_type_triples:
            cons = triple[0]
            typ = triple[2]
            pair_idx = pair2index[(constant2index[cons], constant2index[cons])]
            typ_idx = type2index[typ]
            features[pair_idx][typ_idx] = 1

        for triple in test_graph_real_triples:
            sub = triple[0]
            rel = triple[1]
            obj = triple[2]
            sub_idx = constant2index[sub]
            rel_idx = relation2index[rel]
            obj_idx = constant2index[obj]

            if sub_idx == obj_idx:
                continue

            try:
                pair_idx = pair2index[(sub_idx, obj_idx)]
            except:
                pair_idx = pair2index[(obj_idx, sub_idx)]
                rel_idx = rel_idx + len(relations)
            features[pair_idx][len(types) + rel_idx] = 1

        features.requires_grad = True
        labels.requires_grad = False

        return features, labels, masks

    features, labels, masks = initialize(test_graph_real_triples, test_graph_type_triples,
                                         test_fact_real_triples_with_label, test_fact_type_triples_with_label)

    num_type = len(types)
    num_relation = len(relations)

    print("Finished generation")

    print("Total time elapsed for encoding: {:.4f}s".format(time.time() - t_start))

    return edge, torch.from_numpy(edges).t(), adj, features, labels, masks, len(
        constants), num_type, num_relation, constants, relations, types, pairs, hits_true, r_hits_candidates
