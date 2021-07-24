from __future__ import absolute_import
from __future__ import absolute_import, division, print_function
from __future__ import division
from __future__ import print_function
import glob
import os
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tf_agents.environments import suite_gym, wrappers
from tf_agents.policies.policy_saver import PolicySaver

from gym_game.ai.ai import *
from tf_agents.environments import tf_py_environment
from absl import app
from absl import flags
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image

import tensorflow as tf

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import gym
import gym_game
from gym_game.ai.dqn import DQNWrapper
from gym_game.envs.game_controls import GameControls
from gym_game.envs.game_env import CatchMeIfYouCanEnv
from tensorflow.keras.callbacks import TensorBoard

# Want to play yourself? set to True ;)
prediction = True
# Want to play yourself? set to True ;)
human_mode = True
# train the bot with real arm(True) or with simulation(False)
real_academy = True
# train or test
train = False

#prediction_method = './gym_game/ai/prediction/lstm_100-Dense_5-Step-1_latency_robot_input'
prediction_method = './gym_game/ai/prediction/lstm_100-Dense_5-Step-1_latency_robot_input'
saver_name = "ROBO_500x500_rb-1M_Adam-InverseTD_a-242_n-20_secure"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def del_img():
    files = glob.glob('./img/*')
    for f in files:
        os.remove(f)


def init_folders(saver_name):
    try:
        os.mkdir("./chkpt/chkpt_{}".format(saver_name))
    except:
        pass
    try:
        os.mkdir("./pol/pol_{}".format(saver_name))
    except:
        pass
    try:
        os.mkdir("./train/cqn_{}".format(saver_name))
    except:
        pass


def save():
    os.system("ffmpeg -f image2 -i ./img/image%4d.jpg ./mov/movie.mp4 -y")
    del_img()


def run_episodes_and_create_video(policy, training_env):
    num_episodes = 1
    video_filename = 'imageio.mp4'
    training_env.reset()
    training_env.render()
    save_screen = make_video(training_env.pyenv.envs[0].gym.pygame, training_env.pyenv.envs[0].gym.screen)
    for _ in range(num_episodes):
        time_stepp = training_env.reset()
        for i in range(500):
            action_step = manolete.agent.policy.action(time_stepp)
            time_stepp = training_env.step(action_step.action)
            training_env.render()
            next(save_screen)
        save()


def make_video(pygamee, screen):
    _image_num = 0

    while True:
        _image_num += 1
        str_num = "000" + str(_image_num)
        file_name = "img/image" + str_num[-4:] + ".jpg"
        pygamee.image.save(screen, file_name)
        yield


tf.compat.v1.enable_v2_behavior()
training_py_env = CatchMeIfYouCanEnv()
ax = None
fig = None
losses = []

if real_academy:
    gamecontrols = GameControls()
    training_py_env.set_game_controls(gamecontrols)
    training_py_env.set_academy(real_academy)

else:
    eval_py_env = CatchMeIfYouCanEnv()
    eval_env = wrappers.TimeLimit(eval_py_env, 10000)

if human_mode:
    javris_prediction = None
    if prediction:
        javris_prediction = keras.models.load_model('%s' % prediction_method)

    training_py_env.set_javris_prediction(javris_prediction)
    training_py_env.human_mode = human_mode
    training_py_env.reset()
    pygame = training_py_env.pygame
    while True and pygame:
        training_py_env.render()
        event = pygame.event.get()
        if pygame.key.get_pressed()[pygame.K_c]:
            action = 9
        elif pygame.key.get_pressed()[pygame.K_a]:
            action = 1
        elif pygame.key.get_pressed()[pygame.K_d]:
            action = 2
        elif pygame.key.get_pressed()[pygame.K_w]:
            action = 3
        elif pygame.key.get_pressed()[pygame.K_s]:
            action = 4
        elif pygame.key.get_pressed()[pygame.K_q]:
            action = 5
        elif pygame.key.get_pressed()[pygame.K_y]:
            action = 6
        elif pygame.key.get_pressed()[pygame.K_x]:
            action = 7
        elif pygame.key.get_pressed()[pygame.K_e]:
            action = 8
        elif pygame.key.get_pressed()[pygame.K_ESCAPE]:
            done = True
            break
        else:
            action = 0
        time_step, reward, done, info = training_py_env.step(action)

else:
    init_folders(saver_name)
    # Tensor-Flow-it
    training_py_env = wrappers.TimeLimit(training_py_env, 2400)
    training_env = tf_py_environment.TFPyEnvironment(training_py_env)
    manolete = Manolete(training_env)
    returns = []
    collect_episodes_per_epoch = 100

    train_dir = './train/cqn_{}/'.format(saver_name)
    eval_dir = './eval/cqn_{}/'.format(saver_name)

    train_summary_writer = tf.summary.create_file_writer(
        train_dir, flush_millis=10000)
    eval_summary_writer = tf.summary.create_file_writer(
        eval_dir, flush_millis=10000)
    train_summary_writer.set_as_default()

    saver = PolicySaver(manolete.agent.policy, batch_size=None)
    train_checkpointer = common.Checkpointer(
        ckpt_dir="chkpt/chkpt_{}/".format(saver_name),
        max_to_keep=10,
        agent=manolete.agent,
        policy=manolete.agent.policy,
        replay_buffer=manolete.replay_buffer,
        global_step=manolete.global_step
    )
    train_checkpointer.initialize_or_restore()
    # Dataset generates trajectories with shape [Bx2x...]
    dataset = manolete.replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=manolete.batch_size,
        num_steps=21,
        #num_steps=41,
        single_deterministic_pass=False).prefetch(3)

    iterator = iter(dataset)

    # Add an observer that adds to the replay buffer:
    replay_observer = [manolete.replay_buffer.add_batch]

    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        training_env,
        manolete.agent.collect_policy,
        observers=replay_observer + manolete.train_metrics,
        num_steps=manolete.initial_collect_steps)

    train_driver = dynamic_step_driver.DynamicStepDriver(
        training_env,
        manolete.agent.collect_policy,
        observers=replay_observer + manolete.train_metrics,
        num_steps=1)

    eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        training_env,
        manolete.agent.policy,
        observers=replay_observer + manolete.eval_metrics,
        num_episodes=manolete.num_eval_episodes)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    manolete.agent.train = common.function(manolete.agent.train)
    train_driver.run = common.function(train_driver.run)
    eval_driver.run = common.function(eval_driver.run)
    initial_collect_driver.run = common.function(initial_collect_driver.run)
    # Collect a few steps using collect_policy and save to the replay buffer.
    if manolete.replay_buffer.num_frames() == 0:
        # Collect initial replay data.
        logging.info(
            'Initializing replay buffer by collecting experience for %d steps '
            'with a random policy.', manolete.initial_collect_steps)
        initial_collect_driver.run()


    def train_step():
        exp, _ = next(iterator)
        return manolete.agent.train(exp)


    train_step = common.function(train_step)

    if train:
        for i in range(manolete.global_step.numpy(), manolete.num_iterations + 1):
            train_driver.run()
            train_loss = train_step()
            ##
            step = manolete.global_step.numpy()
            manolete.maybe_log(step, train_loss)
            manolete.pass_train_metrics()

    #        if step % manolete.eval_interval == 0:
    #            eval_summary_writer.set_as_default()
    #            eval_driver.run()
    #            manolete.pass_eval_metrics()
    #            train_summary_writer.set_as_default()
            if manolete.global_step.numpy() % 1000 == 0:
                train_checkpointer.save(global_step=manolete.global_step.numpy())

            if manolete.global_step.numpy() % 50000 == 0:
                saver.save('./pol/pol_{}/policy_{}'.format(saver_name, i))

    else:
        eval_driver.run()
        manolete.pass_eval_metrics()
    #            avg_return = compute_avg_return(training_env, manolete.agent.policy, manolete.num_eval_episodes)
    #            print('step = {0}: Average Return = {1}'.format(step, avg_return))
    #            returns.append(avg_return)
    #            try:
    #                iterations = range(0, manolete.global_step.numpy() + 1, 50000)
    #                plt.ion()
    #                if fig is None:
    #                    fig = plt.figure()
    #                    ax = fig.add_subplot(111)
    #                line = ax.plot(iterations, returns)
    #                fig.canvas.draw()
    #                plt.ylabel('Average Return')
    #                plt.xlabel('Iterations')#
    #
    #            except:
    #                pass

    try:
        iterations = range(0, manolete.num_iterations + 1, manolete.eval_interval)
        plt.plot(iterations, returns)
        plt.ylabel('Average Return')
        plt.xlabel('Iterations')
        # plt.ylim(top=250)
    except:
        print("couldt show chart")
        # iterations = range(495000, 1000001, 1000)
        # max_idx = returns.index(max(returns))
        # shifted = returns[max_idx:] + returns[0:max_idx]
        # diff = []
        # for i in range(len(returns)):
        #    diff.append(round((returns[i] - shifted[i]) / 10000))
        # diff.pop(0)
        # plt.plot(iterations, diff)
        # plt.ylabel('NEW Hits per Iteration')
        # plt.xlabel('Iterations')

    run_episodes_and_create_video(manolete.policy, training_env)

    training_env.close()
    if gamecontrols:
        gamecontrols.on_close(gamecontrols.ws)
        gamecontrols.ws.close()
        del gamecontrols

#FLAGS = flags.FLAGS

# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
#flags.DEFINE_boolean('human_mode', False, 'Set to True to play yourself')
#flags.DEFINE_boolean('real_academy', True, 'Set to True to play with the real robots or false to play with a Simulation')
#flags.DEFINE_boolean('prediction', False, 'Set to True to reduce the latency of the game through Javris-Prediction')

#def main(argv):


#if __name__ == '__main__':
#  app.run(main)