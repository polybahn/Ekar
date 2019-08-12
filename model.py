# 2. we need an LSTM module, input action, output state vector s_t. One at a time
# LSTM module.
from tensorflow.python.keras import Model, Sequential, Input
from tensorflow.python.keras.layers import Input, Dense, LSTM, Embedding, concatenate, Lambda, dot, Softmax, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from load_data import  DataLoader
import numpy as np
import tensorflow as tf
import os
import operator
np.random.seed(123)
tf.random.set_seed(123)


class Ekar(object):

    def __init__(self, keep_rate, data_loader):
        self.batch_size = 1
        self.emb_dim = 32
        self.path_length = 3    # here we define the path length everytime we sample. It's the T in the original paper
        self.state_emb_size = 20
        self.action_emb_size = self.emb_dim * 2    # the input is always a concatenation of [relation, entity] embedding, thus size of 2 x emb_size
        self.beam_width = 64

        self.num_users = data_loader.num_users
        self.num_items = data_loader.num_items
        self.num_relas = data_loader.num_relas
        self.num_embs = self.num_users + self.num_items

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

        self.val_dict = data_loader.val_dict
        self.test_dict = data_loader.test_dict


    def build_model(self):
        rela_input = Input(batch_shape=(self.batch_size, 1), name="relation_input")
        node_input = Input(batch_shape=(self.batch_size, 1), name="node_input")

        hidden_cell_input = Input(batch_shape=(self.batch_size, self.state_emb_size), name="hidden_input")
        state_cell_input = Input(batch_shape=(self.batch_size, self.state_emb_size), name="state_cell_input")

        # # Generate some random weights
        # rela_embedding_weights = np.random.rand(num_relas, emb_dim)
        # node_embedding_weights = np.random.rand(num_vocab, emb_dim)

        rela_embedding_layer = Embedding(
                 input_dim=self.num_relas+1,
                 output_dim=self.emb_dim,
                 trainable=False,
                 name='relation_embedding_layer',
                 weights=[self.rel_emb_matrix],
                 batch_input_shape=[self.batch_size, 1]) (rela_input)
        node_embedding_layer = Embedding(
                 input_dim=self.num_embs+1,
                 output_dim=self.emb_dim,
                 trainable=False,
                 name='node_embedding_layer',
                 weights=[self.node_emb_matrix],
                 batch_input_shape=[self.batch_size, 1]) (node_input)

        # get the embedding for each action
        concat_layer = tf.keras.layers.concatenate(inputs=[rela_embedding_layer, node_embedding_layer], name="rela_node_concat", axis=-1)

        lstm_layer, hidden_state, cell_state = LSTM(units=self.state_emb_size,
                          name="LSTM",
                          return_sequences=True,
                          return_state=True,
                          stateful=False,
                          recurrent_initializer='glorot_uniform')(inputs=concat_layer, initial_state=[hidden_cell_input, state_cell_input])

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
                              ) (policy_layer)

        # Now we define another module to get embedding for all activities
        all_rela_input = Input(batch_shape=(self.batch_size, None), name="all_possible_relation_input")
        all_node_input = Input(batch_shape=(self.batch_size, None), name="all_possible_node_input")

        all_rela_embedding_layer = Embedding(
                 input_dim=self.num_relas+1,
                 output_dim=self.emb_dim,
                 trainable=False,
                 name='all_relation_embedding_layer',
                 weights=[self.rel_emb_matrix],
                 batch_input_shape=[self.batch_size, None]) (all_rela_input)

        all_node_embedding_layer = Embedding(
                 input_dim=self.num_embs+1,
                 output_dim=self.emb_dim,
                 trainable=False,
                 name='all_node_embedding_layer',
                 weights=[self.node_emb_matrix],
                 batch_input_shape=[self.batch_size, None]) (all_node_input)

        all_concat_layer = tf.keras.layers.concatenate(inputs=[all_rela_embedding_layer, all_node_embedding_layer], name="all_rela_node_concat", axis=-1)

        # Next we time all a_t's with y_t
        a_y_dot_layer = tf.squeeze(dot(inputs=[all_concat_layer, hidden_policy_y], axes=-1, name="a_y_dot"), axis=-1)

        # Then finally get the softmax probability
        softmax_layer = Softmax(axis=-1, name="softmax")(a_y_dot_layer)

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
        #     masked_arr = ma.masked_array(possible_states, mask=mask)
        return masked_arr

    def train_one_path(self, target_user_general_id):
        self.model.trainable = True
        cur_rela = self.num_relas - 1
        cur_node = target_user_general_id  # here we sample one user form the training set

        # every time a new path comes in, we need to reset model initial state s_0
        # self.model.reset_states()

        hidden_cell_states = tf.zeros((self.batch_size, self.state_emb_size))
        state_cell_states = tf.zeros((self.batch_size, self.state_emb_size))

        gradBuffer = self.model.trainable_variables
        grads_memory = list()

        # clear grad buffer
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        path = [(cur_rela, cur_node)]
        policy_memory = list()
        loss_per_step = list()
        for i in range(self.path_length):
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
                next_hidden_states, next_cell_states, probabilities = self.model(inputs=[tf.expand_dims([cur_rela], 0),
                                                                               tf.expand_dims([cur_node], 0),
                                                                               tf.expand_dims(next_rels, 0),
                                                                               tf.expand_dims(next_nodes, 0),
                                                                               hidden_cell_states,
                                                                               state_cell_states])
                hidden_cell_states = next_hidden_states
                state_cell_states = next_cell_states

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
                gradBuffer[ix] += tf.multiply(reward, grad)

        # apply gradients
        self.optimizer.apply_gradients(zip(gradBuffer, self.model.trainable_variables))
        return path, policy_memory, loss_per_step


    def beam_search(self, user_id):
        self.model.trainable = False

        class Path(object):
            def __init__(self, path, probability, hidden_state, cell_state):
                self.path = path
                self.probability = probability
                self.hidden_state = hidden_state
                self.cell_state = cell_state

            def __str__(self):
                return ','.join([str(self.path), str(self.probability)])

            def __repr__(self):
                return ','.join([str(self.path), str(self.probability)])

        init_action = (self.num_relas - 1, user_id)
        current_hidden_state = tf.zeros((self.batch_size, self.state_emb_size))
        current_cell_state = tf.zeros((self.batch_size, self.state_emb_size))

        cur_paths = [Path([init_action], 1, current_hidden_state, current_cell_state)]

        for i in range(self.path_length):
            next_paths = list()
            for path in cur_paths:
                cur_rela, cur_node = path.path[-1]

                # infer probability of next actions expanded by this node
                next_actions = np.array(self.lookup_next[cur_node], dtype=np.int32)
                next_rels, next_nodes = tf.unstack(next_actions, axis=-1)
                next_hidden_state, next_cell_state, probabilities = self.model(inputs=[
                    tf.expand_dims([cur_rela], 0),
                    tf.expand_dims([cur_node], 0),
                    tf.expand_dims(next_rels, 0),
                    tf.expand_dims(next_nodes, 0),
                    current_hidden_state,
                    current_cell_state
                ])
                current_hidden_state = next_hidden_state
                current_cell_state = next_cell_state
                probabilities = probabilities.numpy()
                next_paths += [Path(path.path + [(next_action[0], next_action[1])], prob, current_hidden_state, current_cell_state) for next_action, prob in zip(next_actions.tolist(), probabilities[-1].tolist())]
                # sort next_paths by probability
                next_paths.sort(key=operator.attrgetter("probability"), reverse=True)
                if len(next_paths) > self.beam_width:
                    next_paths = next_paths[:self.beam_width]
            cur_paths = next_paths

        # print(cur_paths[:10])
        return cur_paths

    def sigmoid_similarity_lookup_phi(self, first_node, second_node):
        if (first_node, second_node) in self.reward_table:
            return self.reward_table[(first_node, second_node)]
        return .0

    def get_reward(self, target_user_id, destination_node):
        """return the reward for a certain path in graph g' which starts from target_user and ends with desitination_node"""
        if (target_user_id, destination_node) not in self.reward_table:
            return -1.0
        return self.reward_table[(target_user_id, destination_node)]

    def sample_a_user(self):
        return np.random.choice(list(self.positive_rewards.keys()), 1)[-1]

    def get_optimizer(self):
        return Adam(learning_rate=0.001, clipnorm=1.)

    # @tf.function
    def train_one_step(self):
        # first sample one target user
        user_id = self.sample_a_user()
        # then sample a path of this user
        sampled_path, policy_memory, loss_per_step = self.train_one_path(user_id)
        # print(sampled_path)
        # print(policy_memory)
        return np.mean(loss_per_step)

    def evaluate(self, hit_n):
        accumulated_prob = .0
        for user, items in self.val_dict.items():
            output_paths = self.beam_search(user)
            last_items = [path.path[-1][-1] for path in output_paths][:hit_n] # only take top n item to evaluate
            hit_prob = len([i for i in last_items if i in items]) / float(hit_n)
            accumulated_prob += hit_prob
        avg_hit_prob = accumulated_prob / float(len(self.val_dict))
        print(avg_hit_prob)







if __name__=="__main__":
    np.random.seed(3115)
    tf.random.set_seed(3115)

    data_loader = DataLoader()
    keep_rate=0.8
    Ekar = Ekar(keep_rate, data_loader)

    epoch_num = 10
    node_number = 13771

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    Ekar.evaluate(hit_n=10)

    # Training
    num_path = 0
    cumulative_loss = .0
    epoch = 0
    while True:
        if epoch == epoch_num:
            break

        if num_path == node_number:
            print("number path sampled: %d" % num_path)
            print("averaged loss: %f" % (cumulative_loss/num_path))
            Ekar.model.save_weights(checkpoint_prefix.format(epoch=epoch))
            epoch += 1
            cumulative_loss = .0
            num_path = 0
        step_loss = Ekar.train_one_step()
        cumulative_loss += step_loss
        num_path += 1

    Ekar.evaluate(hit_n=10)

