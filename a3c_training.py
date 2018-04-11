import os
import multiprocessing
import threading
import time
import tensorflow as tf
from env_wrapper import TorcsWrapper
from config import Config
from network import Network
from worker import Worker

def main():

    conf = Config()

    tf.reset_default_graph()

    if not os.path.exists(conf.model_path):
        os.makedirs(conf.model_path)

    # Create a directory to save episode playback gifs to
    # if not os.path.exists('./frames'):
    #     os.makedirs('./frames')

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        master_network = Network(conf.state_size, conf.action_size, 'global', None)
        worker_num = multiprocessing.cpu_count()
        workers = []
        for i in range(worker_num):
            workers.append(Worker(i, i, conf.state_size, conf.action_size, trainer, conf.model_path, global_episodes))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if conf.load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(conf.model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(conf.max_steps, conf.max_steps_per_episode, conf.max_episodes, conf.gamma, sess, coord, saver)
            thread = threading.Thread(target=(worker_work))
            thread.start()
            time.sleep(0.5)
            worker_threads.append(thread)
        coord.join(worker_threads)

if __name__ == "__main__":
    main()