from .rl_base import RLBase
import ray
from ray.tune import Trainable, TrainingResult, register_trainable, run_experiments, Experiment


class RLBaseTrainable(Trainable):
    def _setup(self):
        alg = self.config['rl_alg']

        # self.saver = tf.train.Saver()
        self.sess = ...
        self.iteration = 0

    def _train(self):
        self.sess.run(...)
        self.iteration += 1

    def _save(self, checkpoint_dir):
        return self.saver.save(
            self.sess, checkpoint_dir + "/save",
            global_step=self.iteration)

    def _restore(self, path):
        return self.saver.restore(self.sess, path)