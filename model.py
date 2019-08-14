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
import os
import random
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)



class Ekar(object):

    def __init__(self, batch_size, keep_rate, data_loader):
        self.batch_size = batch_size
        self.emb_dim = 32
        self.path_length = 3    # here we define the path length everytime we sample. It's the T in the original paper
        self.state_emb_size = 20
        self.action_emb_size = self.emb_dim * 2    # the input is always a concatenation of [relation, entity] embedding, thus size of 2 x emb_size

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
        cur_relas = [self.num_relas - 1] * batch_size
        cur_nodes = target_user_ids

        hidden_cell_states = tf.zeros((self.batch_size, self.state_emb_size))
        state_cell_states = tf.zeros((self.batch_size, self.state_emb_size))

        # self.model.reset_states()

        gradBuffer = self.model.trainable_variables
        grads_memory = list()

        # clear grad buffer
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        # path = [[item] for item in list(zip(cur_relas, cur_nodes))]
        # # policy_memory = list()
        loss_memory = list()

        for i in range(self.path_length):
            next_actions = [self.lookup_next[cur_node] for cur_node in cur_nodes]
            next_actions = self.mask_batch_next_actions(next_actions, self.keep_rate)
            len_actions = [len(next_action) for next_action in next_actions]

            next_rels = tf.keras.preprocessing.sequence.pad_sequences([[action[0] for action in actions] for actions in next_actions], padding='post')#, maxlen=self.max_out_degree)
            next_nodes = tf.keras.preprocessing.sequence.pad_sequences([[action[1] for action in actions] for actions in next_actions], padding='post')#, maxlen=self.max_out_degree)

            cur_relas = np.array(cur_relas, dtype=np.int32)
            cur_nodes = np.array(cur_nodes, dtype=np.int32)

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.model.trainable_variables)
                # get probability of all of nexrt actions
                next_hidden_states, next_cell_states, probabilities = self.model(inputs=[tf.expand_dims(cur_relas, -1),
                                                   tf.expand_dims(cur_nodes, -1),
                                                   next_rels,
                                                   next_nodes,
                                                   hidden_cell_states,
                                                   state_cell_states])

                # for next hidden states and cell states
                hidden_cell_states = next_hidden_states
                state_cell_states = next_cell_states

                # now we remove all the invalid action probabilities
                sampled_ids = list()
                for probability in probabilities.numpy():
                    try:
                        sampled_ids.append(np.random.choice(len(probability), size=1, p=probability)[-1])
                    except ValueError:
                        print(probability)

                gathered_probs = tf.gather_nd(probabilities, indices=list(enumerate(sampled_ids)))
                # compute loss
                batch_loss = -tf.math.log(gathered_probs)
                # print(loss)
                loss_memory.append(batch_loss)

                # this has to stay inside the scope of Tape
                grads = [tape.gradient(single_loss, self.model.trainable_variables) for single_loss in batch_loss]
            grads_memory.append(grads)

            cur_relas = [next_rel[index] for index, next_rel in zip(sampled_ids, next_rels)]
            cur_nodes = [next_node[index] for index, next_node in zip(sampled_ids, next_nodes)]

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
        return tf.reduce_mean(loss_memory).numpy()





    def train_one_path(self, path_length, target_user_general_id):
        #     self.model.trainable = False
        cur_rela = self.num_relas - 1
        cur_node = target_user_general_id  # here we sample one user form the training set

        # every time a new path comes in, we need to reset model initial state s_0
        self.model.reset_states()
        gradBuffer = self.model.trainable_variables
        grads_memory = list()

        # clear grad buffer
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        path = [(cur_rela, cur_node)]
        policy_memory = list()
        loss_per_step = list()
        for i in range(path_length):
            # get all id for next state
            next_actions = np.array(self.lookup_next[cur_node], dtype=np.int32)
            #         print(next_actions.shape)
            # actions after masking
            next_actions = self.mask_next_actions(next_actions, self.keep_rate)
            num_next_options = next_actions.shape[0]
            #         print(next_actions.shape)
            # extend action space length to maximum number of out degree

            next_rels, next_nodes = tf.unstack(next_actions,
                                               axis=-1)  # unzip the actions to make rela_list and node_list
            #         print(next_rels)
            #         print(tf.expand_dims(next_nodes, 0))

            with tf.GradientTape() as tape:
                # get probability of all of nexrt actions
                probabilities = self.model(inputs=[tf.expand_dims([cur_rela], 0),
                                              tf.expand_dims([cur_node], 0),
                                              tf.expand_dims(next_rels, 0),
                                              tf.expand_dims(next_nodes, 0),])
            #    print(probabilities)
                # sample one id for next actions
                # sampled_id = tf.random.categorical(probabilities, num_samples=1)[-1, 0].numpy()
                sampled_id = np.random.choice(num_next_options, size=1, p=probabilities[-1].numpy())[-1]
                # compute loss
                loss = - tf.math.log(probabilities[-1][sampled_id])
                loss_per_step.append(loss.numpy())
            # compute gradients in this step
            grads = tape.gradient(loss, self.model.trainable_variables)
            grads_memory.append(grads)

            # now we have the input for next step
            policy_memory.append(probabilities[-1][sampled_id].numpy())
            # update current relation and node for entering next step
            cur_rela, cur_node = next_actions[sampled_id]
            path.append((cur_rela, cur_node))

        # get reward of this path
        dest_node = path[-1][-1]  # the destination node of this path
        reward = self.get_reward(target_user_general_id, dest_node)
        # calculate gradients
        for grads in grads_memory:
            for ix, grad in enumerate(grads):
                gradBuffer[ix] += reward * grad

        # apply gradients
        self.optimizer.apply_gradients(zip(gradBuffer, self.model.trainable_variables))
        return path, policy_memory, loss_per_step

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
    def train_one_step(self):
        # first sample one target user
        user_ids = self.sample_batch_users()
        # then sample a path of this user
        # sampled_path, policy_memory, loss_per_step = \

        self.train_batch_paths(user_ids)
        # print(sampled_path)
        # print(policy_memory)
        return 0


if __name__=="__main__":
    data_loader = DataLoader()
    keep_rate = 0.8
    batch_size = 512
    Ekar = Ekar(batch_size, keep_rate, data_loader)

    Ekar.train_one_step()

# if __name__=="__main__":
#
#     data_loader = DataLoader()
#     keep_rate=0.8
#     batch_size = 512
#     Ekar = Ekar(batch_size, keep_rate, data_loader)
#
#     epoch_num = 300000
#     node_number = 13771
#
#     # Directory where the checkpoints will be saved
#     checkpoint_dir = './training_checkpoints'
#     # Name of the checkpoint files
#     checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
#
#     num_path = 0
#     cumulative_loss = .0
#     epoch = 0
#     while True:
#         if epoch == epoch_num:
#             break
#
#         if num_path == node_number:
#             print("number path sampled: %d" % num_path)
#             print("averaged loss: %f" % (cumulative_loss/num_path))
#             Ekar.model.save_weights(checkpoint_prefix.format(epoch=epoch))
#             epoch += 1
#             cumulative_loss = .0
#             num_path = 0
#         step_loss = Ekar.train_one_step()
#         cumulative_loss += step_loss
#         num_path += 1
