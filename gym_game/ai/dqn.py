from __future__ import absolute_import, division, print_function

import base64
import time

import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay

import tensorflow as tf
from tensorflow.python.keras.callbacks_v1 import TensorBoard

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


class DQNWrapper:
    def __init__(self, training_env):
        self.NAME = "DQN-500x500".format(int(time.time()))
        tensorboard = TensorBoard(log_dir="logs/{}".format(self.NAME))

        self.training_env = training_env
        self.num_iterations = 1000000  # @param {type:"integer"}
        self.gamma = 0.99
        self.initial_collect_steps = 5000  # @param {type:"integer"}
        self.collect_steps_per_iteration = 1  # @param {type:"integer"}
        self.replay_buffer_max_length = 500000  # @param {type:"integer"}

        self.batch_size = 100  # @param {type:"integer"}
        self.learning_rate = 1e-4  # @param {type:"number"}
        self.log_interval = 200  # @param {type:"integer"}

        self.num_eval_episodes = 1  # @param {type:"integer"}
        self.eval_interval = 5000  # @param {type:"integer"}

        self.fc_layer_params = (500, 500)
        self.action_tensor_spec = tensor_spec.from_spec(self.training_env.action_spec())
        self.num_actions = self.action_tensor_spec.maximum - self.action_tensor_spec.minimum + 1

        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # it's output.
        self.dense_layers = [self.dense_layer(num_units) for num_units in self.fc_layer_params]
        self.q_values_layer = tf.keras.layers.Dense(
            self.num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        model = tf.keras.models.Sequential([
            self.dense_layers,
            self.q_values_layer
        ])
        self.q_net = sequential.Sequential(self.dense_layers + [self.q_values_layer])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        self.agent = dqn_agent.DqnAgent(
            self.training_env.time_step_spec(),
            self.training_env.action_spec(),
            q_network=self.q_net,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.global_step)
        self.agent.initialize()
        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.training_env.batch_size,
            max_length=self.replay_buffer_max_length)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(self, num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    def create_model(self, env):
        n_actions = env.action_space.n
        obs_shape = env.observation_space.shape
        observations_input = tf.keras.layers.Input(obs_shape, name='observations_input')
        action_mask = tf.keras.layers.Input((n_actions,), name='action_mask')
        hidden = tf.keras.layers.Dense(500, activation='relu')(observations_input)
        hidden_2 = tf.keras.layers.Dense(500, activation='relu')(hidden)
        output = tf.keras.layers.Dense(n_actions)(hidden_2)
        filtered_output = tf.keras.layers.multiply([output, action_mask])
        model = tf.keras.models.Model([observations_input, action_mask], filtered_output)
        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate, clipnorm=1.0)
        model.compile(optimizer, loss='mean_squared_error')
        return model
