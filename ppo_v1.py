import os
# *IMPORTANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(66)

import gym
import copy
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import highway_env
import cv2


def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(66)
   tf.compat.v2.random.set_seed(66)
   tf.random.set_seed(66)
   np.random.seed(66)


# make some random data
reset_random_seeds()

tfd = tfp.distributions
tf.keras.backend.set_floatx('float64')

def model(state_shape, action_dim, units=(128, 128)):
    state = Input(shape=state_shape)

    vf = Dense(units[0], name="Value_L0", activation="tanh")(state)
    for index in range(1, len(units)):
        vf = Dense(units[index], name="Value_L{}".format(index), activation="tanh")(vf)

    value_pred = Dense(1, name="Out_value")(vf)

    pi = Dense(units[0], name="Policy_L0", activation="tanh")(state)
    for index in range(1, len(units)):
        pi = Dense(units[index], name="Policy_L{}".format(index), activation="tanh")(pi)

    action_probs = Dense(action_dim, name="Out_probs", activation='softmax')(pi)
    model = Model(inputs=state, outputs=[action_probs, value_pred])

    return model


class PPO:
    def __init__(self, env, out_dir,
            lr=5e-4, hidden_units=(128, 128), c1=1.0, c2=0.01,
            clip_ratio=0.2, gamma=0.95, lam=1.0, batch_size=128, n_updates=4):
        self.env = env
        self.out_dir = out_dir
        self.state_shape = int(np.prod(env.observation_space.shape))
        self.action_dim = env.action_space.n

        # Define and initialize network
        self.policy = model(self.state_shape, self.action_dim, hidden_units)
        self.model_optimizer = Adam(learning_rate=lr)
        print(self.policy.summary())

        # Set hyperparameters
        self.gamma = gamma  # discount factor
        self.lam = lam
        self.c1 = c1  # value difference coeff
        self.c2 = c2  # entropy coeff
        self.clip_ratio = clip_ratio  # for clipped surrogate
        self.batch_size = batch_size
        self.n_updates = n_updates  # number of epochs per episode
        self.summaries = {}

    def get_dist(self, output):
        dist = tfd.Categorical(probs=output)
        return dist

    def evaluate_actions(self, state, action):
        output, value = self.policy(state)
        dist = self.get_dist(output)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return log_probs, entropy, value

    def act(self, state, test=False):
        state = np.expand_dims(state, axis=0).astype(np.float64)
        output, value = self.policy.predict(state)
        dist = self.get_dist(output)
        action = tf.math.argmax(output, axis=-1) if test else dist.sample()
        log_probs = dist.log_prob(action)
        return action[0].numpy(), value[0][0], log_probs[0].numpy()

    def save_model(self, fn):
        self.policy.save(fn)

    def load_model(self, fn):
        self.policy.load_weights(fn)
        print(self.policy.summary())

    def get_gaes(self, rewards, v_preds, next_v_preds):
        # source: https://github.com/uidilr/ppo_tf/blob/master/ppo.py#L98
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, next_v_preds, v_preds)]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.lam * self.gamma * gaes[t + 1]
        return gaes

    def learn(self, observations, actions, log_probs, next_v_preds, rewards, gaes):
        rewards = np.expand_dims(rewards, axis=-1).astype(np.float64)
        next_v_preds = np.expand_dims(next_v_preds, axis=-1).astype(np.float64)

        with tf.GradientTape() as tape:
            new_log_probs, entropy, state_values = self.evaluate_actions(observations, actions)

            ratios = tf.exp(new_log_probs - log_probs)
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1-self.clip_ratio,
                                              clip_value_max=1+self.clip_ratio)
            loss_clip = tf.minimum(gaes * ratios, gaes * clipped_ratios)
            loss_clip = tf.reduce_mean(loss_clip)

            target_values = rewards + self.gamma * next_v_preds
            vf_loss = tf.reduce_mean(tf.math.square(state_values - target_values))

            entropy = tf.reduce_mean(entropy)
            total_loss = -loss_clip + self.c1 * vf_loss - self.c2 * entropy

        train_variables = self.policy.trainable_variables
        grad = tape.gradient(total_loss, train_variables)  # compute gradient
        self.model_optimizer.apply_gradients(zip(grad, train_variables))

        # tensorboard info
        self.summaries['total_loss'] = total_loss
        self.summaries['surr_loss'] = loss_clip
        self.summaries['vf_loss'] = vf_loss
        self.summaries['entropy'] = entropy

    def train(self, max_epochs=8000, max_steps=100, save_freq=50):
        train_log_dir = self.out_dir + '/logs/'
        summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.epoch = 0

        while self.epoch < max_epochs:
            done, steps = False, 0
            cur_state = self.env.reset()
            obs, actions, log_probs, rewards, v_preds, next_v_preds = [], [], [], [], [], []

            while not done:
                action, value, log_prob = self.act(cur_state)
                next_state, reward, done, _ = self.env.step(action)
                # self.env.render(mode='rgb_array')

                rewards.append(reward)
                v_preds.append(value)
                obs.append(cur_state)
                actions.append(action)
                log_probs.append(log_prob)

                steps += 1
                cur_state = next_state

            next_v_preds = v_preds[1:] + [0]
            gaes = self.get_gaes(rewards, v_preds, next_v_preds)
            gaes = np.array(gaes).astype(dtype=np.float64)
            gaes = (gaes - gaes.mean()) / gaes.std()
            data = [obs, actions, log_probs, next_v_preds, rewards, gaes]

            # Sample training data
            sample_indices = np.arange(len(rewards))
            sampled_data = [np.take(a=a, indices=sample_indices, axis=0) for a in data]

            # Train model
            self.learn(*sampled_data)

            # Tensorboard update
            with summary_writer.as_default():
                tf.summary.scalar('Loss/total_loss', self.summaries['total_loss'], step=self.epoch)
                tf.summary.scalar('Loss/clipped_surr', self.summaries['surr_loss'], step=self.epoch)
                tf.summary.scalar('Loss/vf_loss', self.summaries['vf_loss'], step=self.epoch)
                tf.summary.scalar('Loss/entropy', self.summaries['entropy'], step=self.epoch)

            summary_writer.flush()

            self.epoch += 1
            print("epochs #{}: reward {}, steps {} ".format(self.epoch, np.sum(rewards), steps))

            # Tensorboard update
            with summary_writer.as_default():
                tf.summary.scalar('Main/episode_reward', np.sum(rewards), step=self.epoch)
                tf.summary.scalar('Main/episode_steps', steps, step=self.epoch)

            summary_writer.flush()

            if steps >= max_steps:
                print("episode {}, reached max steps".format(self.epoch))
                self.save_model(self.out_dir + "/ppo_episode{}.h5".format(self.epoch))

            if self.epoch % save_freq == 0:
                self.save_model(self.out_dir + "/ppo_episode{}.h5".format(self.epoch))
                # self.test()

        self.save_model(self.out_dir + "/ppo_final_episode{}.h5".format(self.epoch))

    def test(self, render=True, fps=5):
        cur_state, done, rewards = self.env.reset(), False, 0

        rendered_frame = self.env.render(mode="rgb_array")
        video_filename = os.path.join(output_dir, 'videos/',"testing_episode_{}".format(self.epoch) +
                                      '.mp4')
        # Init video recording
        if video_filename is not None:
            print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape,
                                                                  fps))
            video_recorder = VideoRecorder(video_filename,
                                           frame_size=rendered_frame.shape, fps=fps)
            video_recorder.add_frame(rendered_frame)
        else:
            video_recorder = None

        while not done:
            action, value, log_prob = self.act(cur_state, test=True)
            next_state, reward, done, _ = self.env.step(action)
            cur_state = next_state
            rewards += reward
            if render:
                rendered_frame = self.env.render(mode="rgb_array")
                if video_recorder is not None:
                    video_recorder.add_frame(rendered_frame)
        if video_recorder is not None:
            video_recorder.release()
        return rewards


class VideoRecorder:
    """This is used to record videos of evaluations"""

    def __init__(self, filename, frame_size, fps):
        self.video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"MPEG"), int(fps),
            (frame_size[1], frame_size[0]))

    def add_frame(self, frame):
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()


if __name__ == "__main__":
    gym_env = gym.make("CartPole-v1")  # CartPole-v1, highway-v0
    if not os.path.exists("results"):
        os.mkdir("results")
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")
    output_dir = "results/" + now
    os.makedirs(output_dir, exist_ok = True)
    os.makedirs(output_dir + "/videos", exist_ok = True)
    ppo = PPO(gym_env, out_dir=output_dir)

    # ppo.load_model("ppo_final_episode250.h5")
    ppo.train(max_epochs=10000, save_freq=200)
    reward = ppo.test()
    print("Total rewards: ", reward)