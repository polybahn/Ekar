# 2. we need an LSTM module, input action, output state vector s_t. One at a time
# LSTM module.
from tensorflow.python.keras import Model, Sequential, Input
from tensorflow.python.keras.layers import InputLayer, Dense, LSTM, Embedding, concatenate, Lambda, dot, Softmax, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.backend as KB
from load_data import DataLoader
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import operator

import os
import random
random.seed(3115)
np.random.seed(3115)
tf.random.set_seed(3115)



class Ekar(object):

    def __init__(self, batch_size, keep_rate, data_loader):
        self.batch_size = batch_size
        self.emb_dim = 32
        self.path_length = 3    # here we define the path length everytime we sample. It's the T in the original paper
        self.state_emb_size = 60
        self.action_emb_size = self.emb_dim * 2    # the input is always a concatenation of [relation, entity] embedding, thus size of 2 x emb_size
        self.beam_width = 64


        self.num_users = data_loader.num_users
        self.num_items = data_loader.num_items
        self.num_relas = data_loader.num_relas
        self.num_embs = self.num_users + self.num_items

        self.max_out_degree = data_loader.max_out_degree

        self.node_emb_matrix = data_loader.node_emb_matrix
        self.rel_emb_matrix = data_loader.rel_emb_matrix
        for i in range(self.node_emb_matrix.shape[0]):
            for j in range(self.node_emb_matrix.shape[1]):
                if np.isinf(self.node_emb_matrix[i][j]):
                    print("nan!")
                    print(self.node_emb_matrix[i][j])


        self.model = self.build_model()
        self.optimizer = self.get_optimizer()

        self.lookup_next = data_loader.lookup_next
        self.reward_table = data_loader.reward_table
        self.positive_rewards = data_loader.positive_rewards
        self.keep_rate = keep_rate

        self.train_set = data_loader.train_set
        self.val_dict = data_loader.val_dict
        self.test_dict = data_loader.test_dict


    def lookup(self, key):
        if isinstance(key, tf.Tensor):
            key = key.numpy()
        if key == 0:
            return []
        return self.lookup_next[key]

    def build_model(self):
        rela_input = Input(shape=(1,), batch_size=self.batch_size, dtype=tf.int32, name="relation_input")
        node_input = Input(shape=(1,), batch_size=self.batch_size, dtype=tf.int32, name="node_input")

        hidden_cell_input = Input(batch_shape=(self.batch_size, self.state_emb_size), name="hidden_input")
        state_cell_input = Input(batch_shape=(self.batch_size, self.state_emb_size), name="state_cell_input")

        # # Generate some random weights
        # rela_embedding_weights = np.random.rand(num_relas, emb_dim)
        # node_embedding_weights = np.random.rand(num_vocab, emb_dim)

        rela_embedding_layer = Embedding(
                 input_dim=self.num_relas+1,
                 output_dim=self.emb_dim,
                 trainable=False,
                 mask_zero=True,
                 name='relation_embedding_layer',
                 weights=[self.rel_emb_matrix])(rela_input)
        node_embedding_layer = Embedding(
                 input_dim=self.num_embs+1,
                 output_dim=self.emb_dim,
                 trainable=False,
                 mask_zero=True,
                 name='node_embedding_layer',
                 weights=[self.node_emb_matrix],
                 )(node_input)

        # get the embedding for each action
        concat_layer = tf.keras.layers.concatenate(inputs=[rela_embedding_layer, node_embedding_layer], name="rela_node_concat", axis=-1)

        lstm_layer, hidden_state, cell_state = LSTM(units=self.state_emb_size,
                                                    name="LSTM",
                                                    return_sequences=True,
                                                    return_state=True,
                                                    stateful=False,
                                                    recurrent_initializer='glorot_uniform')(inputs=concat_layer,
                                                                                            initial_state=[hidden_cell_input,
                                                                                                           state_cell_input])

        # policy module
        dense_unit_size = 128
        policy_layer = Dense(units=dense_unit_size,
                               name="first_dense_with_relu",
                               kernel_initializer="glorot_uniform",
                               bias_initializer="zeros",
                               activation="relu")(lstm_layer)
        # This is y_t
        hidden_policy_y = Dense(units=self.action_emb_size,
                               name="second_dense_without_activation",
                               kernel_initializer="glorot_uniform",
                               bias_initializer="zeros",
                               activation=None
                              )(policy_layer)

        # Now we define another module to get embedding for all activities
        all_rela_input = Input(shape=(None,), batch_size=self.batch_size, dtype=tf.int32, name="all_possible_relation_input")
        all_node_input = Input(shape=(None,), batch_size=self.batch_size, dtype=tf.int32, name="all_possible_node_input")

        all_rela_embedding_layer = Embedding(
                 input_dim=self.num_relas+1,
                 output_dim=self.emb_dim,
                 mask_zero=True,
                 trainable=False,
                 name='all_relation_embedding_layer',
                 weights=[self.rel_emb_matrix])(all_rela_input)

        all_node_embedding_layer = Embedding(
                 input_dim=self.num_embs+1,
                 output_dim=self.emb_dim,
                 mask_zero=True,
                 trainable=False,
                 name='all_node_embedding_layer',
                 weights=[self.node_emb_matrix])(all_node_input)

        all_concat_layer = tf.keras.layers.concatenate(inputs=[all_rela_embedding_layer, all_node_embedding_layer], name="all_rela_node_concat", axis=-1)

        # Next we time all a_t's with y_t
        a_y_dot_layer = tf.squeeze(tf.keras.layers.Dot(axes=-1)(inputs=[all_concat_layer, hidden_policy_y]), axis=-1, name="a_y_dot")

        def custom_activation(x):
            x = KB.switch(tf.math.is_nan(x), KB.zeros_like(x), x)  # prevent nan values
            x = KB.switch(KB.equal(KB.exp(x), 1), KB.zeros_like(x), KB.exp(x))
            return x / KB.sum(x, axis=-1, keepdims=True)

        # Then finally get the softmax probability
        softmax_layer = tf.keras.layers.Activation(custom_activation)(a_y_dot_layer)

        model = Model(inputs=[rela_input, node_input, all_rela_input, all_node_input, hidden_cell_input, state_cell_input], outputs=[hidden_state, cell_state, softmax_layer], name="ekar")

        model.summary()
        return model

    def mask_next_actions(self, possible_states, keep_rate):
        num_keep = np.math.floor(len(possible_states) * keep_rate)
        if num_keep == 0:
            num_keep = 1
        mask = np.zeros(len(possible_states), dtype=np.int32)
        mask[:num_keep] = 1
        np.random.shuffle(mask)
        masked_arr = np.array([action for action, m in zip(possible_states, mask) if m])  # to make size compatible
        return masked_arr

    def mask_batch_next_actions(self, list_possible_states, keep_rate):
        num_keeps = [max(np.math.floor(len(possible_states) * keep_rate), 1) for possible_states in list_possible_states]
        masks = [[1] * num_keep + [0] * (len(possible_states) - num_keep)for num_keep, possible_states in zip(num_keeps, list_possible_states)]
        for mask in masks:
            random.shuffle(mask)
        masked = [[state for m, state in zip(mask, possible_states) if m] for mask, possible_states in zip(masks, list_possible_states)]
        return masked


    def train_batch_paths(self, target_user_ids):
        self.model.trainable = True
        cur_relas = np.array([self.num_relas - 1] * batch_size, dtype=np.int32)
        cur_nodes = np.array(target_user_ids, dtype=np.int32)

        hidden_cell_states = tf.zeros((self.batch_size, self.state_emb_size))
        state_cell_states = tf.zeros((self.batch_size, self.state_emb_size))

        gradBuffer = self.model.trainable_variables
        grads_memory = list()

        # clear grad buffer
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        loss_memory = list()

        for i in range(self.path_length):
            next_actions = [self.lookup(cur_node) for cur_node in cur_nodes]
            next_actions = self.mask_batch_next_actions(next_actions, self.keep_rate)
            # len_actions = [len(next_action) for next_action in next_actions]
            next_rels = [[rel for rel, node in actions] for actions in next_actions]
            next_nodes = [[node for rel, node in actions] for actions in next_actions]
            next_rels = tf.keras.preprocessing.sequence.pad_sequences(next_rels, padding='post')
            next_nodes = tf.keras.preprocessing.sequence.pad_sequences(next_nodes, padding='post')

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.model.trainable_variables)
                # get probability of all of nexrt actions
                next_hidden_states, next_cell_states, probabilities = self.model(inputs=[tf.expand_dims(cur_relas, -1),
                                                   tf.expand_dims(cur_nodes, -1),
                                                   next_rels,
                                                   next_nodes,
                                                   hidden_cell_states,
                                                   state_cell_states])
                # now we remove all the invalid action probabilities
                sampled_ids = [tfp.distributions.Categorical(probs=probability).sample().numpy() for probability in probabilities]
                # for next hidden states and cell states
                hidden_cell_states = next_hidden_states
                state_cell_states = next_cell_states

                gathered_probs = tf.gather_nd(probabilities, indices=list(enumerate(sampled_ids)))
                # compute loss
                batch_loss = -tf.math.log(gathered_probs)
                # print(loss)
                loss_memory.append(batch_loss)

                # this has to stay inside the scope of Tape
                grads = [tape.gradient(single_loss, self.model.trainable_variables) for single_loss in batch_loss]
            grads_memory.append(grads)

            cur_relas = tf.gather_nd(next_rels, indices=list(enumerate(sampled_ids)))
            cur_nodes = tf.gather_nd(next_nodes, indices=list(enumerate(sampled_ids)))

            # cur_relas = [next_rel[index] for index, next_rel in zip(sampled_ids, next_rels)]
            # cur_nodes = [next_node[index] for index, next_node in zip(sampled_ids, next_nodes)]

        rewards = self.get_rewards(target_user_ids, cur_nodes)

        # finalize gradient by multiplying with reward respectively
        for grads in grads_memory:
            for i, reward in zip(range(self.batch_size), rewards):
                for j in range(len(grads[0])):
                    grads[i][j] = tf.multiply(grads[i][j], reward)

        # update gradBuffer
        for i in range(self.batch_size):
            for grads in grads_memory:
                for ix, grad in enumerate(grads[i]):
                    gradBuffer[ix] += grad

        # normalize gradients
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad / self.batch_size

        # apply gradients
        self.optimizer.apply_gradients(zip(gradBuffer, self.model.trainable_variables))
        print(rewards)
        print(loss_memory)
        return tf.reduce_mean(loss_memory).numpy()


    def beam_search(self, user_ids):
        self.model.trainable = False

        cur_relas = [self.num_relas - 1] * batch_size
        cur_nodes = user_ids

        cur_time_paths = tf.convert_to_tensor([[(rel, node)] for rel, node in zip(cur_relas, cur_nodes)], dtype=tf.int32)


        cur_time_probability = None
        cur_time_hidden_cell_states = tf.zeros([self.batch_size, 1, self.state_emb_size], dtype=tf.float32) #[batch, beam, emb_size]
        cur_time_state_cell_states = tf.zeros([self.batch_size, 1, self.state_emb_size], dtype=tf.float32) #[batch, beam, emb_size]
        cur_time_paths = tf.reshape(cur_time_paths, [self.batch_size, 1, 1, 2])  #[batch, beam, cur_len, (edge, node)]
        for i in range(self.path_length):
            # we maintain one current path and possibility
            next_level_paths = tf.zeros([self.batch_size, self.beam_width, i+2, 2], dtype=tf.int32)
            next_probabilities = tf.zeros([self.batch_size, self.beam_width], dtype=tf.float32)
            next_level_hidden = tf.zeros([self.batch_size, self.beam_width, self.state_emb_size], dtype=tf.float32)
            next_level_cell = tf.zeros([self.batch_size, self.beam_width, self.state_emb_size], dtype=tf.float32)

            for slice_id in range(cur_time_paths.shape[1]):
                cur_paths = tf.slice(cur_time_paths, begin=[0, slice_id, 0, 0], size=[self.batch_size, 1, -1, 2])
                cur_actions = tf.reshape(tf.slice(cur_time_paths, begin=[0, slice_id, i, 0], size=[self.batch_size, 1, -1, 2]), [self.batch_size, 2])
                # print(cur_actions.shape)
                hidden_cell_states = tf.reshape(tf.slice(cur_time_hidden_cell_states, begin=[0, slice_id, 0], size=[self.batch_size, 1, -1]), [self.batch_size, self.state_emb_size])
                state_cell_states = tf.reshape(tf.slice(cur_time_state_cell_states, begin=[0, slice_id, 0], size=[self.batch_size, 1, -1]), [self.batch_size, self.state_emb_size])

                cur_rels, cur_nodes = tf.unstack(cur_actions, axis=-1)
                next_actions = [self.lookup(cur_node) for cur_node in cur_nodes]
                max_next_len = max([len(acts) for acts in next_actions])
                next_rels = tf.keras.preprocessing.sequence.pad_sequences(
                    [[action[0] for action in actions] for actions in next_actions],
                    padding='post', maxlen=max(max_next_len, self.beam_width))
                next_nodes = tf.keras.preprocessing.sequence.pad_sequences(
                    [[action[1] for action in actions] for actions in next_actions],
                    padding='post', maxlen=max(max_next_len, self.beam_width))

                next_hidden_states, next_cell_states, probabilities = self.model(inputs=[tf.expand_dims(cur_relas, -1),
                                                                                         tf.expand_dims(cur_nodes, -1),
                                                                                         next_rels,
                                                                                         next_nodes,
                                                                                         hidden_cell_states,
                                                                                         state_cell_states])
                hidden_cell_states = next_hidden_states
                state_cell_states = next_cell_states

                # print(hidden_cell_states.shape)
                # print(probabilities.shape)
                top_k_probs, top_k_indices = tf.math.top_k(probabilities, k=self.beam_width)
                # print(top_k_probs.shape)
                nd_indices = [[[i, indice] for indice in indices] for i, indices in enumerate(top_k_indices)]
                next_rels = tf.gather_nd(next_rels, indices=nd_indices)
                next_nodes = tf.gather_nd(next_nodes, indices=nd_indices)
                # stack to form the next actions
                next_actions = tf.reshape(tf.stack([next_rels, next_nodes], axis=-1), [self.batch_size, self.beam_width, 1, 2])
                # print(next_actions.shape)

                # get next paths
                cur_paths = tf.tile(cur_paths, [1, self.beam_width, 1, 1])
                # print(cur_paths.shape)
                cur_paths = tf.concat([cur_paths, next_actions], axis=-2)
                # print(cur_paths.shape)

                # get next hiddens and cells
                hidden_cell_states = tf.tile(hidden_cell_states, [1, self.beam_width])
                hidden_cell_states = tf.reshape(hidden_cell_states, [self.batch_size, self.beam_width, self.state_emb_size])
                # print(hidden_cell_states.shape)
                state_cell_states = tf.tile(state_cell_states, [1, self.beam_width])
                state_cell_states = tf.reshape(state_cell_states, [self.batch_size, self.beam_width, self.state_emb_size])
                # print(state_cell_states.shape)

                # append current solution to existing ones
                next_level_paths = tf.concat([next_level_paths, cur_paths], axis=1)
                next_probabilities = tf.concat([next_probabilities, top_k_probs], axis=1)
                next_level_hidden = tf.concat([next_level_hidden, hidden_cell_states], axis=1)
                next_level_cell = tf.concat([next_level_cell, state_cell_states], axis=1)
                # print(next_level_paths.shape)

                # reshape next level data
                next_probabilities, top_k_indices = tf.math.top_k(next_probabilities, k=self.beam_width)
                nd_indices = [[[i, indice] for indice in indices] for i, indices in enumerate(top_k_indices)]
                # print(next_probabilities.shape)
                next_level_paths = tf.gather_nd(next_level_paths, indices=nd_indices)
                next_level_hidden = tf.gather_nd(next_level_hidden, indices=nd_indices)
                next_level_cell = tf.gather_nd(next_level_cell, indices=nd_indices)
                # print(next_level_paths.shape)
                # print(next_level_hidden.shape)
                # print(next_level_cell.shape)
            # update cur_path
            cur_time_paths = next_level_paths
            cur_time_hidden_cell_states = next_level_hidden
            cur_time_state_cell_states = next_level_cell
            cur_time_probability = next_probabilities
            # print("cur_time_paths" + str(cur_time_paths.shape))

        return cur_time_paths, cur_time_probability

    def sigmoid_similarity_lookup_phi(self, first_node, second_node):
        if (first_node, second_node) in self.reward_table:
            return self.reward_table[(first_node, second_node)]
        return .0

    def get_rewards(self, target_user_ids, destination_ids):
        return [self.get_reward(user, dest) for user, dest in zip(target_user_ids, destination_ids)]

    def get_reward(self, target_user_id, destination_node):
        """return the reward for a certain path in graph g' which starts from target_user and ends with desitination_node"""
        if (target_user_id, destination_node) not in self.reward_table:
            return - 1.0
        return self.reward_table[(target_user_id, destination_node)]

    def sample_a_user(self):
        return np.random.choice(list(self.positive_rewards.keys()), 1)[-1]

    def sample_batch_users(self):
        return np.random.choice(list(self.positive_rewards.keys()), self.batch_size).tolist()

    def get_optimizer(self):
        return Adam(learning_rate=0.001, clipnorm=1.)

    # @tf.function
    def train_one_step(self, user_ids):
        # first sample one target user
        user_ids = self.sample_batch_users()
        # then sample a path of this user
        self.train_batch_paths(user_ids)
        return 0

    def evaluate(self, ndcg_n):
        # compute perfect possible dcg
        idcg = np.sum([1/np.log2(i+2) for i in range(ndcg_n)])

        accumulated_prob = .0
        accumulated_ndcg = .0

        valid_users = tf.data.Dataset.from_tensor_slices(list(self.val_dict.keys()))
        user_sequences = valid_users.batch(self.batch_size, drop_remainder=True)
        valid_items = list(self.val_dict.values())
        item_sequences = [valid_items[i*self.batch_size:i*self.batch_size+self.batch_size] for i in range(int(len(valid_items)//self.batch_size))]
        print(len(item_sequences))

        hit_accumulator = .0
        ndcg_accumulator = .0
        cnt_total = .0
        for user_batch, item_batch in zip(user_sequences, item_sequences):
            output_paths, probabilities = self.beam_search(user_batch)
            print(output_paths.shape)
            # print(probabilities)
            last_nodes = tf.reshape(tf.slice(output_paths, begin=[0, 0, self.path_length, 1], size=[self.batch_size, ndcg_n, 1, 1]), [self.batch_size, ndcg_n]).numpy()
            # print(last_nodes)
            batch_hits = np.sum([np.sum([1 if item in positive_items else 0 for item in top_n_items])/ndcg_n for positive_items, top_n_items in zip(item_batch, last_nodes)])
            batch_dcg = [np.sum([1/np.log2(i+2)if item in positive_items else 0 for i, item in enumerate(top_n_items)]) for positive_items, top_n_items in zip(item_batch, last_nodes)]
            batch_idcg = [np.sum([1/np.log2(i+2) for i in range(len(positive_items))]) if len(positive_items) < ndcg_n else idcg for positive_items in item_batch]
            batch_ndcg = sum([s_dcg/s_idcg for s_dcg, s_idcg in zip(batch_dcg, batch_idcg)])

            hit_accumulator += batch_hits
            ndcg_accumulator += batch_ndcg
            cnt_total += self.batch_size
        avg_hits_prob = hit_accumulator / cnt_total
        avg_ndcg_prob = ndcg_accumulator / cnt_total
        return avg_hits_prob, avg_ndcg_prob




if __name__=="__main__":
    data_loader = DataLoader()
    keep_rate = 0.8
    batch_size = 64
    ndcg_n = 10
    ekar = Ekar(batch_size, keep_rate, data_loader)

    BUFFER_SIZE = 10000
    training_epoch = 10

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    if os.path.exists(checkpoint_dir):
        ekar.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    # Creating training dataset
    train_users = [u for u, _, _ in ekar.train_set]
    node_number = len(train_users)
    print("total training nodes per epoch:\t%d" % node_number)
    train_users = tf.data.Dataset.from_tensor_slices(train_users)\
        .repeat(training_epoch)\
        .shuffle(BUFFER_SIZE)\
        .batch(batch_size, drop_remainder=True)


    # Training
    batch_trained = 0
    epoch = 0
    batches_per_epoch = int(node_number//batch_size)
    train_loss_accumulator = .0
    for batch_users in train_users:
        print(batch_users)
        batch_train_loss = ekar.train_batch_paths(batch_users.numpy())
        batch_trained += 1
        print("training loss at batch %d:\t%f" % (batch_trained, batch_train_loss))
        train_loss_accumulator += batch_train_loss
        # Evaluation
        if batch_trained % batches_per_epoch == 0:
            epoch += 1
            print("average training loss at epoch %d:\t%.5f" % (epoch, train_loss_accumulator/batches_per_epoch))
            train_loss_accumulator = .0
            ekar.model.save_weights(checkpoint_prefix.format(epoch=epoch))
            hr_10, ndcg_10 = ekar.evaluate(ndcg_n)
            print("averaged hit rate at %d:\t%.5f " % (ndcg_n, hr_10))
            print("averaged NDCG at %d:\t%.5f " % (ndcg_n, ndcg_10))


