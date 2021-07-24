import os

import tensorflow as tf
from absl import logging

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics, py_metric
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    environment.render()
    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


def compute_avg_return(env, policy, num_episodes=5):
    total_return = 0.0
    for i_episode in range(num_episodes):
        time_step = env.reset()
        done = False
        episode_return = 0.0

        print("{}: {}".format("Episode", i_episode))
        #        for t in range(1000):
        while not time_step.is_last():
            action = policy.action(time_step)
            time_step = env.step(action)
            episode_return += time_step.reward
            env.render()
            if done:
                break
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


class Manolete:

    # https://github.com/tensorflow/agents/blob/master/tf_agents/agents/dqn/examples/v2/train_eval.py
    def __init__(self, training_env, checkpoint=None):
        tf.compat.v1.enable_v2_behavior()
        self.training_env = training_env
        self.num_iterations = 1000000  # @param {type:"integer"}
        # self.num_iterations = 15000  # @param {type:"integer"}
        self.initial_collect_steps = 10000  # @param {type:"integer"}
        self.collect_steps_per_iteration = 1  # @param {type:"integer"}
        #self.replay_buffer_capacity = 500000  # @param {type:"integer"}
        self.replay_buffer_capacity = 1000000  # @param {type:"integer"}
        # 2 hidden layers with 100 and 50 neurons
        self.hidden_layers = (500, 500)
        self.steps_per_epoch = 2400
        # if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.
        self.batch_size = 100  # @param {type:"integer"}
        self.learning_rate = 1e-5  # @param {type:"number"}
        self.gamma = 0.99
        self.log_interval = 200  # @param {type:"integer"}
        self._summary_interval = 1000
        self.num_atoms = 242  # @param {type:"integer"}
        #self.num_atoms = 250  # @param {type:"integer"}
        self.min_q_value = -10  # @param {type:"integer"}
        #self.min_q_value = 0  # @param {type:"integer"}
        self.max_q_value = 110  # @param {type:"integer"}
        #self.max_q_value = 2500  # @param {type:"integer"}
        self.n_step_update = 20  # @param {type:"integer"}
        #self.n_step_update = 40  # @param {type:"integer"}

        self.num_eval_episodes = 10  # @param {type:"integer"}
        self.eval_interval = 50000  # @param {type:"integer"}}

        self.categorical_q_net = categorical_q_network.CategoricalQNetwork(
            self.training_env.observation_spec(),
            self.training_env.action_spec(),
            num_atoms=self.num_atoms,
            fc_layer_params=self.hidden_layers)

        # The code below sets a schedules.InverseTimeDecay to hyperbolically decrease the learning rate to 1/2 of the
        # base rate at 1000 epochs, 1/3 at 2000 epochs and so on.
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=self.steps_per_epoch * 1000,
            #decay_steps=self.steps_per_epoch * 100,
            decay_rate=1,
            staircase=False)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # self.optimizer = tf.compat.v1.train.RMSPropOptimizer(
        #    learning_rate=self.learning_rate,
        #    decay=0.95,
        #    momentum=0.0,
        #    epsilon=0.00001,
        #    centered=True)

        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        self.agent = categorical_dqn_agent.CategoricalDqnAgent(
            self.training_env.time_step_spec(),
            self.training_env.action_spec(),
            categorical_q_network=self.categorical_q_net,
            optimizer=self.optimizer,
            min_q_value=self.min_q_value,
            max_q_value=self.max_q_value,
            n_step_update=self.n_step_update,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=self.gamma,
            train_step_counter=self.global_step)
        self.agent.initialize()
        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy

        self.policy = random_tf_policy.RandomTFPolicy(self.training_env.time_step_spec(),
                                                      self.training_env.action_spec())
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.training_env.batch_size,
            max_length=self.replay_buffer_capacity)

        self.train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=1),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=self.num_eval_episodes),
        ]
        self.eval_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=1),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=self.num_eval_episodes)
        ]

    def maybe_record_summaries(self, global_step_val):
        """Record summaries if global_step_val is a multiple of summary_interval."""
        if global_step_val % self._summary_interval == 0:
            py_metric.run_summaries(self.train_metrics)

    def maybe_log(self, global_step_val, total_loss):
        """Log some stats if global_step_val is a multiple of log_interval."""
        if global_step_val % self.log_interval == 0:
            logging.info('step = %d, loss = %f', global_step_val, total_loss.loss)
            for metric in self.train_metrics:
                self.log_metric(metric, prefix='Train/Metrics')

    def log_metric(self, metric, prefix):
        tag = common.join_scope(prefix, metric.name)
        logging.info('%s', '{0} = {1}'.format(tag, metric.result()))

    def pass_train_metrics(self):
        for train_metric in self.train_metrics:
            train_metric.tf_summaries(
                train_step=self.global_step, step_metrics=self.train_metrics[:2])

    def pass_eval_metrics(self):
        for eval_metric in self.eval_metrics:
            eval_metric.tf_summaries(
                train_step=self.global_step, step_metrics=self.eval_metrics[:2])
