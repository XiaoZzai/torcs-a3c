import numpy as np
import tensorflow as tf
from env_wrapper import TorcsWrapper
from utils import update_target_graph, discount
from network import Network

class Worker():
    def __init__(self, index, name, state_size, action_size, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        # Local network
        self.local_network = Network(state_size, action_size, self.name, trainer)

        # Update operations
        self.update_local_ops = update_target_graph('global', self.name)

        # # The Below code is related to setting up the Doom environment
        # game.set_doom_scenario_path("basic.wad")  # This corresponds to the simple task we will pose our agent
        # game.set_doom_map("map01")
        # game.set_screen_resolution(ScreenResolution.RES_160X120)
        # game.set_screen_format(ScreenFormat.GRAY8)
        # game.set_render_hud(False)
        # game.set_render_crosshair(False)
        # game.set_render_weapon(True)
        # game.set_render_decals(False)
        # game.set_render_particles(False)
        # game.add_available_button(Button.MOVE_LEFT)
        # game.add_available_button(Button.MOVE_RIGHT)
        # game.add_available_button(Button.ATTACK)
        # game.add_available_game_variable(GameVariable.AMMO2)
        # game.add_available_game_variable(GameVariable.POSITION_X)
        # game.add_available_game_variable(GameVariable.POSITION_Y)
        # game.set_episode_timeout(300)
        # game.set_episode_start_time(10)
        # game.set_window_visible(False)
        # game.set_sound_enabled(False)
        # game.set_living_reward(-1)
        # game.set_mode(Mode.PLAYER)
        # game.init()
        # self.actions = self.actions = np.identity(a_size, dtype=bool).tolist() # ?
        # End Doom set-up
        self.env = TorcsWrapper(port=3101+index)

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = np.asarray([ _ for _ in rollout[:, 0]])
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = np.asarray([ _ for _ in rollout[:, 3]])
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_network.target_v: discounted_rewards,
                     self.local_network.state_input: observations,
                     self.local_network.actions: actions,
                     self.local_network.advantages: advantages,
                     self.local_network.state_in[0]: self.batch_rnn_state[0],
                     self.local_network.state_in[1]: self.batch_rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([self.local_network.value_loss,
                                                                     self.local_network.policy_loss,
                                                                     self.local_network.entropy,
                                                                     self.local_network.grad_norms,
                                                                     self.local_network.var_norms,
                                                                     self.local_network.state_out,
                                                                     self.local_network.apply_grads],
                                                                     feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_steps, max_steps_per_episode, max_episodes, gamma, sess, coord, saver):
        try:
            episodes = sess.run(self.global_episodes)
            total_steps = 0
            print("Starting worker " + str(self.number))
            with sess.as_default(), sess.graph.as_default():
                while not coord.should_stop():
                    if episodes >= max_episodes:
                        break

                    sess.run(self.update_local_ops)
                    episode_buffer = []
                    episode_values = []
                    # episode_frames = []
                    episode_reward = 0
                    steps_per_episode = 0

                    state_t = self.env.reset()
                    # s = self.env.get_state().screen_buffer
                    # episode_frames.append(s)
                    # s = process_frame(s)
                    rnn_state = self.local_network.state_init
                    self.batch_rnn_state = rnn_state
                    while steps_per_episode < max_steps_per_episode:
                        # Take an action using probabilities from policy network output.
                        actor_policy, critic_value, rnn_state = sess.run(
                            [self.local_network.policy, self.local_network.value, self.local_network.state_out],
                            feed_dict={self.local_network.state_input: [state_t],
                                       self.local_network.state_in[0]: rnn_state[0],
                                       self.local_network.state_in[1]: rnn_state[1]})

                        if np.random.random() < 0.2:
                            a = np.random.randint(0, 5)
                        else:
                            a = np.argmax(actor_policy[0])

                        state_t1, reward, done = self.env.step(a)
                        print("%s, Step=%d, Action=%d" % (self.name, steps_per_episode, a))
                        # if done == False:
                        #     episode_frames.append(s1)
                        #     s1 = process_frame(s1)
                        # else:
                        #     s1 = s

                        episode_buffer.append([state_t, a, reward, state_t1, done, critic_value[0, 0]])
                        episode_values.append(critic_value[0, 0])

                        episode_reward += reward
                        state_t = state_t1
                        total_steps += 1
                        steps_per_episode += 1

                        # If the episode hasn't ended, but the experience buffer is full, then we
                        # make an update step using that experience rollout.
                        if len(episode_buffer) == 30 and done == False and steps_per_episode != max_steps_per_episode - 1:
                            # Since we don't know what the true final return is, we "bootstrap" from our current
                            # value estimation.
                            v1 = sess.run(self.local_network.value,
                                          feed_dict={self.local_network.state_input: [state_t],
                                                     self.local_network.state_in[0]: rnn_state[0],
                                                     self.local_network.state_in[1]: rnn_state[1]})[0, 0]
                            v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                            episode_buffer = []
                            sess.run(self.update_local_ops)
                        if done == True:
                            break

                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(steps_per_episode)
                    self.episode_mean_values.append(np.mean(episode_values))

                    # Update the network using the episode buffer at the end of the episode.
                    if len(episode_buffer) > 0:
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                    # # Periodically save gifs of episodes, model parameters, and summary statistics.
                    # if episode_count % 5 == 0 and episode_count != 0:
                    #     if self.name == 'worker_0' and episode_count % 25 == 0:
                    #         time_per_step = 0.05
                    #         images = np.array(episode_frames)
                    #         make_gif(images, './frames/image' + str(episode_count) + '.gif',
                    #                  duration=len(images) * time_per_step, true_image=True, salience=False)
                    #     if episode_count % 250 == 0 and self.name == 'worker_0':
                    #         saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                    #         print("Saved Model")
                    #
                    #     mean_reward = np.mean(self.episode_rewards[-5:])
                    #     mean_length = np.mean(self.episode_lengths[-5:])
                    #     mean_value = np.mean(self.episode_mean_values[-5:])
                    #     summary = tf.Summary()
                    #     summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    #     summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    #     summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    #     summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    #     summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    #     summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    #     summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    #     summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    #     self.summary_writer.add_summary(summary, episode_count)
                    #
                    #     self.summary_writer.flush()

                    if self.name == 'worker_0':
                        sess.run(self.increment)

                    episodes += 1
        finally:
            self.env.end()