import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tf_agents.networks import q_network  # Q net
from tf_agents.agents.dqn import dqn_agent  # DQN Agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer  # replay buffer
from tf_agents.trajectories import trajectory  # s->s' trajectory
from tf_agents.utils import common  # loss function
from tf_agents.policies import policy_saver

logger = logging.getLogger("root")


class DQN:
    """
    Deep Q-Network
    """
    def __init__(
        self,
        train_env,
        eval_env,
        learning_rate,
        replay_buffer_max_length,
        batch_size,
        num_iterations,
        num_eval_episodes,
        collect_steps_per_iteration,
    ):

        self.train_env = train_env
        self.eval_env = eval_env

        self.learning_rate = learning_rate
        self.replay_buffer_max_length = replay_buffer_max_length
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.num_eval_episodes = num_eval_episodes
        self.collect_steps_per_iteration = collect_steps_per_iteration

        self.log_interval = 200
        self.eval_interval = 500

        logger.info(
            "Initializing the Deep Deterministic Policy Gradient class variables"
        )

    def q_net(self):
        fc_layer_params = (100,)

        q_net = q_network.QNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=fc_layer_params,
        )

        return q_net

    # Data Collection
    def collect_step(self, environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)

    def collect_data(self, env, policy, buffer, steps):
        for _ in range(steps):
            self.collect_step(env, policy, buffer)

    # eval_env evaluation
    def compute_avg_return(self, environment, policy, num_episodes=10):

        total_return = 0.0

        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def train(self):

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        train_step_counter = tf.Variable(0)

        agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=self.q_net(),
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter,
        )

        agent.initialize()

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_max_length,
        )

        # Fetch experience
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=self.batch_size, num_steps=2
        ).prefetch(3)

        iterator = iter(dataset)

        # step 5 - training - takes a while
        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        agent.train = common.function(agent.train)

        # Reset the train step
        agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = self.compute_avg_return(
            self.eval_env, agent.policy, self.num_eval_episodes
        )
        # some statistics
        loss_history = {"episode": [], "loss_ex": []}
        return_history = {"episode": [0], "avg_return": [avg_return]}

        for _ in range(self.num_iterations):

            # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(self.collect_steps_per_iteration):
                self.collect_step(self.train_env, agent.collect_policy, replay_buffer)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss

            step = agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                print("step = {0}: loss = {1}".format(step, train_loss))
                loss_history["episode"].append(step)
                loss_history["loss_ex"].append(float(train_loss))
            
            if step % self.eval_interval == 0:
                avg_return = self.compute_avg_return(
                    self.eval_env, agent.policy, self.num_eval_episodes
                )
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                return_history["episode"].append(step)
                return_history["avg_return"].append(avg_return)

        loss_name = os.path.join(
            f"experiments/{os.getenv('EXPERIMENT_NO')}/history", "loss_history.csv"
        )
        df_loss = pd.DataFrame.from_dict(loss_history)
        df_loss.to_csv(loss_name, index=False, encoding="utf-8")

        return_name = os.path.join(
            f"experiments/{os.getenv('EXPERIMENT_NO')}/history", "return_history.csv"
        )
        df_return = pd.DataFrame.from_dict(return_history)
        df_return.to_csv(return_name, index=False, encoding="utf-8")

        policy_dir = f"experiments/{os.getenv('EXPERIMENT_NO')}/greedy_policy"
        tf_policy_saver = policy_saver.PolicySaver(agent.policy)
        tf_policy_saver.save(policy_dir)

        # saved_policy = tf.compat.v2.saved_model.load(policy_dir)

        return agent.policy
