from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
import random
import pickle


class DataLoader(object):
    # def __new__(cls, filepath='loader.pkl', *args, **kwargs):
    #     if os.path.exists(filepath):
    #         with open(filepath, 'rb') as f:
    #             inst = pickle.load(f)
    #         if not isinstance(inst, cls):
    #             raise TypeError('Unpickled object is not of type {}'.format(cls))
    #     else:
    #         inst = super(DataLoader, cls).__new__(cls, *args, **kwargs)
    #     return inst


    def __init__(self):
        loaded_path = 'loader.pkl'
        if os.path.exists(loaded_path):
            with open(loaded_path, 'rb') as f:
                self.__dict__ = pickle.load(f)
            return

        # define some parameter
        self.emb_dim = 32

        # define data path
        self.data_path = "/Users/polybahn/Desktop/MKR/data/movie"
        self.kg_file_name = "kg_final.npy"
        self.rating_file_name = "ratings_final.npy"
        self.item_2_entity_map_name = "item_index2entity_id.txt"

        self.eimap = self.map_entity_to_item()

        # load rating file and kg file
        # [user_id, item_id, rating]
        ratings = np.load(self.get_path(self.rating_file_name))
        # [entity_id, relation_id, entity_id]
        kg = np.load(self.get_path(self.kg_file_name))

        self.entity_space, self.item_space, self.item_ref = self.get_entity_item_space(ratings, kg)
        self.user_space, self.user_ref, self.rela_space = self.create_user_rel_space(ratings, kg)
        self.generate_new_ratings_and_kg(ratings, kg)
        self.get_rev_indexing()
        self.build_action_lookup()
        self.create_reward_table()
        self.node_emb_matrix, self.rel_emb_matrix = self.get_embedding_matrices()

        if not os.path.exists("loader.pkl"):
            with open('loader.pkl', 'wb') as pk_f:
                pickle.dump(self.__dict__, pk_f)




    def get_path(self, file_name):
            return os.path.join(self.data_path, file_name)

    def map_entity_to_item(self):
        # load data
        # define map[entity_id] = item_id
        eimap = dict()
        with open(self.get_path(self.item_2_entity_map_name), 'r') as id_map:
            for line in id_map:
                item_id, entity_id = [int(e) for e in line.strip().split('\t')]
                eimap[entity_id] = item_id
        return eimap

    def get_entity_item_space(self, ratings, kg):
        # We first make sure the entity_id and item_id space the same
        # scan rating list to get all items
        item_space = set()
        for _, i, _ in ratings:
            item_space.add(i)
        for h, _, t in kg:
            if h in self.eimap:
                item_space.add(self.eimap[h])
            if t in self.eimap:
                item_space.add(self.eimap[t])
        # check all entity not as items
        entity_space = set()
        for h, _, t in kg:
            if h not in self.eimap:
                # entity not present as item
                entity_space.add(h)
            if t not in self.eimap:
                entity_space.add(t)

        # represent these entity by a new id
        for e in entity_space:
            temp = e
            while temp in item_space:
                temp += 1
            # now temp is the new id for this entity in item space
            item_space.add(temp)
            # also update reference to the new item
            self.eimap[e] = temp
        # so far the item_space are all items & entities next we create a mapping from item_id -> id in whole graph
        item_ref = dict(zip(item_space, range(len(item_space))))
        return entity_space, item_space, item_ref


    def create_user_rel_space(self, ratings, kg):
        # then we create a mapping from user_id -> new_id
        user_space = set()
        user_ref = dict()
        cur_id = len(self.item_space)
        for u, _, _ in ratings:
            if u not in user_space:
                user_space.add(u)
                user_ref[u] = cur_id
                cur_id += 1

        # # add empty virtual node to fill in the action input matrix
        # node_empty = u
        # while node_empty in user_space:
        #     node_empty += 1
        # user_space.add(u)
        # user_ref[node_empty] = cur_id
        # empty_node_id = cur_id  # this id should equals to len(nodes in the graph)-1
        # cur_id += 1

        # finally we create relationship mapping
        rela_space = set()
        for _, r, _ in kg:
            if r not in rela_space:
                rela_space.add(r)

        # one more thing in mind: name the new relation: interact_with. and self_loop relation
        self.interact_with = r + 1
        while self.interact_with in rela_space:
            self.interact_with += 1
        rela_space.add(self.interact_with)

        # do self loop relation
        self.self_loop_relation = self.interact_with + 1
        while self.self_loop_relation in rela_space:
            self.self_loop_relation += 1
        rela_space.add(self.self_loop_relation)

        return user_space, user_ref, rela_space

    def generate_new_ratings_and_kg(self, ratings, kg):
        # next we generate the new rating file in new G space and add them to G
        self.new_ratings = list()
        self.new_g = list()
        # we have another G' which do not include reverse links and self loops
        conv_g = list()
        for u, i, r in ratings:
            self.new_ratings.append([self.user_ref[u], self.item_ref[i], r])
            # add rating links
            self.new_g.append([self.user_ref[u], self.interact_with, self.item_ref[i]])
            self.new_g.append([self.item_ref[i], self.interact_with, self.user_ref[u]])

            conv_g.append([self.user_ref[u], self.interact_with, self.item_ref[i]])

        # then we further complete the new knowledge_graph
        for ori_h, r, ori_t in kg:
            item_h = self.eimap[ori_h]
            item_t = self.eimap[ori_t]
            # add normal link
            self.new_g.append([self.item_ref[item_h], r, self.item_ref[item_t]])
            conv_g.append([self.item_ref[item_h], r, self.item_ref[item_t]])

            # add reverse link
            self.new_g.append([self.item_ref[item_t], r, self.item_ref[item_h]])

        # add self-loop links to users and items
        for general_user in self.user_ref.values():
            self.new_g.append([general_user, self.self_loop_relation, general_user])
            conv_g.append([general_user, self.self_loop_relation, general_user])
        for general_item in self.item_ref.values():
            self.new_g.append([general_item, self.self_loop_relation, general_item])
            conv_g.append([general_item, self.self_loop_relation, general_item])

        # save the graph G' for preprocessing by ConvE
        random.shuffle(conv_g)
        conv_train = conv_g[:int(len(conv_g) * 0.6)]
        conv_val = conv_g[int(len(conv_g) * 0.6) + 1: int(len(conv_g) * 0.8)]
        conv_test = conv_g[int(len(conv_g) * 0.8) + 1:]
        print("train\t%d\t val \t%d\t test\t%d\t" % (len(conv_train), len(conv_val), len(conv_test)))

        def write_graph(dataset, f_name):
            with open("/Users/polybahn/Desktop/ConvE/data/EKAR_movie/" + f_name + ".txt", "w") as f:
                for h, r, t in dataset:
                    f.write('\t'.join([str(h), str(r), str(t)]))
                    f.write('\n')


        write_graph(conv_train, "train")
        write_graph(conv_val, "valid")
        write_graph(conv_test, "test")

        # We get the size of the embedding space
        self.num_embs = (len(self.item_ref) + len(self.user_ref))
        self.num_users = len(self.user_ref)
        self.num_items = len(self.item_ref)
        self.num_relas = len(self.rela_space)

        # assert the globl id is continuously growing and no missing id in between
        ass_id = [False] * (self.num_users + self.num_items)
        for v in self.item_ref.values():
            ass_id[v] = True
        for v in self.user_ref.values():
            ass_id[v] = True
        for i in ass_id:
            if not i:
                print("assertion not passed. exit")
                exit(-1)

        print("assertion passed.")

        # print some stats
        print("number entities: \t%d" % len(self.eimap))
        print("number users: \t%d" % len(self.user_ref))
        print("number items&entities: \t%d" % len(self.item_ref))
        print("number relations: \t%d" % self.num_relas)

        print("number links in graph: \t%d" % len(self.new_g))
        print("number ratings: \t%d" % len(self.new_ratings))

        print("number of embedding space: \t%d" % self.num_embs)

        # save the global kg and rating file somewhere for future use
        np.save(self.get_path("ekar_kg_final.npy"), self.new_g)
        np.save(self.get_path("ekar_ratings_final.npy"), self.new_ratings)

    def get_rev_indexing(self):
        # now the indexing goes as following:
        # eimap: entity_id -> item_id,  item_ref: item_id -> sequencial_global_id
        # user_ref: user_id -> sequential_global_id
        # Need to create reverse maps to map to original items & users & entities
        self.rev_user_ref = dict([(v, k) for k, v in self.user_ref.items()])
        self.rev_item_ref = dict([(v, k) for k, v in self.item_ref.items()])
        self.rev_eimap = dict([(v, k) for k, v in self.eimap.items()])


    def build_action_lookup(self):
        # Create user set and entity set
        self.all_users = set(self.rev_user_ref.keys())
        self.all_items = set(self.rev_item_ref.keys())
        print("number of users: %d" % len(self.all_users))
        print("number of items: %d" % len(self.all_items))

        # Define the adjacency matrix. Given an node, find the next entity id and next node in Graph G'
        self.lookup_next = dict()
        # for each head we store the next link and corresponding node in a list
        for head, link, tail in self.new_g:
        #     print(','.join([str(rev_rela_ref[link]), str(head), str(tail)]))
            if head not in self.lookup_next:
                self.lookup_next[head] = list()
            self.lookup_next[head].append((link, tail))

        # we get the maximum value of out-degree in G'
        max_out_degree = max([len(v) for v in self.lookup_next.values()])
        print("Maximum value of out-degree in the graph is:\t%d" % max_out_degree)

        # Then we need to generate a ground-truth reward table for every user
        self.positive_rewards = dict()
        for user, item, _ in self.new_ratings:
            # here we don't care about the rating is 0 or 1. If user interacted with the item, then reward is 1
            if user not in self.positive_rewards:
                self.positive_rewards[user] = set()
            self.positive_rewards[user].add(item)

    def load_phi_scores(self):
        path = "/Users/polybahn/Desktop/ConvE/data/full_score.npy"
        scores = np.load(path, allow_pickle=True).item()
        scores_final = dict()
        for key, val in scores.items():
            new_key = (key[0], key[-1])
            old_score = .0
            if new_key in scores_final:
                old_score = scores_final[new_key]
            scores_final[new_key] = max(old_score, val)
        return scores_final

    def create_reward_table(self):
        # load conv-E embeddings
        def lnp_n(d_type):
            # load numpy dictionary
            path = "/Users/polybahn/Desktop/ConvE/data/"
            return np.load(path + d_type + '_node.npy', allow_pickle=True).item()

        self.node_embs = dict()
        self.node_embs.update(lnp_n('full'))

        print(len(self.node_embs.keys()))
        print(list(self.node_embs.values())[0])
        print(list(self.node_embs.values())[0].shape)

        def lnp_r(d_type):
            # load numpy dictionary
            path = "/Users/polybahn/Desktop/ConvE/data/"
            return np.load(path + d_type + '_rel.npy', allow_pickle=True).item()

        self.rel_embs = dict()
        self.rel_embs.update(lnp_r('full'))
        print(self.rel_embs.keys())

        # first we create an lookup table for reward
        invalid_user_num = 0
        invalid_item_num = 0
        self.reward_table = dict()

        # add reward according to phi function
        score_phi = self.load_phi_scores()
        for user in self.all_users:
            user_id = str(user)
            if user_id not in self.node_embs:
                invalid_user_num += 1
                continue
            for item in self.all_items:
                item_id = str(item)
                if item_id not in self.node_embs:
                    invalid_item_num += 1
                    continue
                # add reward
                if (user_id, item_id) in score_phi:
                    self.reward_table[(user, item)] = score_phi[(user_id, item_id)]
                    # print("phi:\t" + str(score_phi[(user_id, item_id)]))
        # add positive reward from the ratings in dataset
        for user, item, _ in self.new_ratings:
            self.reward_table[(user, item)] = 1.0

    def get_embedding_matrices(self):
        # We further constrain our space only with the nodes having ConvE embeddings by initializing embedding matrix with node indexes
        node_emb_matrix = np.zeros([self.num_embs + 1, self.emb_dim], dtype=np.float32)
        for n_id, emb in self.node_embs.items():
            node_emb_matrix[int(n_id)] = emb

        rel_emb_matrix = np.zeros([self.num_relas + 1, self.emb_dim], dtype=np.float32)
        for r_id, emb in self.rel_embs.items():
            if '_' in r_id:
                continue
            rel_emb_matrix[int(r_id)] = emb
        return node_emb_matrix, rel_emb_matrix


if __name__=="__main__":
    loader = DataLoader()
    # check if saved model before:
    # if not os.path.exists("loader.pkl"):
    #     with open('loader.pkl', 'w') as pk_f:
    #         pickle.dump(loader, pk_f)