import tensorflow as tf
from config import Config
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split


def linear(input_, output_size, scope=None):
    """
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class Discriminator:
    def __init__(self, config):
        # configuration
        self.max_len = config["max_len"]
        # topic nums + 1
        self.num_classes = config["n_class"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.filter_sizes = config["filter_sizes"]
        self.num_filters = config["num_filters"]
        self.l2_reg_lambda = config["l2_reg_lambda"]
        self.topic_num = config["topic_num"]
        self.learning_rate = config["learning_rate"]

        # label_smooth  positive : 1 -> alpha
        self.ls_alpha = config["label_smooth"]
        # placeholder
        self.input_x = tf.placeholder(tf.int32, [None, self.max_len], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def build_graph(self, calc_f1=False):
        print("building discriminator graph ... ")
        l2_loss = tf.constant(0.0)
        with tf.variable_scope("discriminator"):
            # Embedding:
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)  # batch_size * seq * embedding_size
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            pooled_outputs = list()
            # Create a convolution + max-pool layer for each filter size
            for filter_size, filter_num in zip(self.filter_sizes, self.num_filters):
                with tf.name_scope("cov2d-maxpool%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, filter_num]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # print(conv.name, ": ", conv.shape) batch * (seq - filter_shape) + 1 * 1(output channel) *
                    # filter_num
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.max_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")  # 全部池化到 1x1
                    # print(conv.name, ": ", conv.shape , "----", pooled.name, " : " ,pooled.shape)
                    pooled_outputs.append(pooled)
            total_filters_num = sum(self.num_filters)

            self.h_pool = tf.concat(pooled_outputs, 3)
            # print(self.h_pool.shape) # batch * 1 * 1 * total_filters_num
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_filters_num])  # batch * total_num

            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # add droppout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([total_filters_num, self.num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.sigmoid(self.scores)


                reward_masked = self.ypred_for_auc * self.input_y
                total_reward = tf.reduce_sum(reward_masked, axis=1)
                label_num = tf.count_nonzero(self.input_y, axis=1, dtype=tf.float32)  # one hot
                # senti GAN :use penalty-based objective  1 - reward
                self.rewards_for_mlc = 1 - tf.divide(total_reward, label_num)
                # origin SeqGAN
                # self.rewards_for_mlc = tf.divide(total_reward, label_num)
                self.predictions = self._multi_label_hot(self.ypred_for_auc)
                self.hamming_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.cast(
                        tf.logical_xor(tf.cast(self.predictions, tf.bool), tf.cast(self.input_y, tf.bool)), tf.float32),
                        axis=1),
                    axis=0)
                # print(self.predictions.shape) # self_size x num_class
            with tf.name_scope("loss"):
                if self.ls_alpha:
                    smooth_target = self.input_y * self.ls_alpha  # may need temperature ?
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=smooth_target)
                # losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
                multi_class_loss = tf.reduce_mean(
                    tf.reduce_sum(losses, axis=1)
                )
                self.loss = multi_class_loss + self.l2_reg_lambda * l2_loss
            with tf.name_scope("accuracy"):
                # print(self.input_y.shape)
                self.accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(self.predictions, self.input_y), tf.float32))
            #  micro-f1  precision and recall
            self.m_p, self.m_r, self.m_f1 = self.micro_f1(self.predictions, self.input_y)

            self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
            d_optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # aggregation_method =2 能够帮助减少内存占用
            grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
            self.train_op = d_optimizer.apply_gradients(grads_and_vars)

        print("discriminator graph successfully built!")

    def _multi_label_hot(self, prediction, name="prediction", threshold=0.7):
        prediction = tf.cast(prediction, tf.float32)
        threshold = float(threshold)
        return tf.cast(tf.greater(prediction, threshold), tf.float32, name=name)


    def micro_f1(self, pred, label):
        tp = tf.reduce_mean(tf.reduce_sum(pred * label, axis=1))
        fn = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.logical_xor(
            tf.cast(pred, tf.bool), tf.cast(label, tf.bool)
        ), tf.float32) * label, axis=1))
        fp = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.logical_xor(
            tf.cast(pred, tf.bool), tf.cast(label, tf.bool)
        ), tf.float32) * pred, axis=1))
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        return p, r, f1

    @staticmethod
    def restore(sess, saver, path):
        saver.restore(sess, save_path=path)
        print("discrminator load successfully!")

    def _make_train_fd(self, x_batch, y_batch):
        return {self.input_x: x_batch,
                self.input_y: y_batch,
                self.dropout_keep_prob: .5}

    def run_train_epoch(self, sess, x_batch, y_batch, fetch_f1=False):
        fd = self._make_train_fd(x_batch, y_batch)
        fetch = [self.train_op, self.loss, self.accuracy, self.hamming_loss]
        if fetch_f1:
            fetch.extend([self.m_f1, self.m_p, self.m_r])
        return sess.run(fetch, feed_dict=fd)

    def _make_test_fd(self, x_batch, y_batch):
        return {self.input_x: x_batch,
                self.input_y: y_batch,
                self.dropout_keep_prob: 1.0}

    def run_test_epoch(self, sess, x_batch, y_batch, fetch_f1=False):
        fd = self._make_test_fd(x_batch, y_batch)
        fetch = [self.accuracy, self.hamming_loss]
        if fetch_f1:
            fetch.extend([self.m_f1, self.m_p, self.m_r])
        return sess.run(fetch, feed_dict=fd)


if __name__ == '__main__':
    # test model
    config = Config()
    model = Discriminator(config.discriminator_config)
    model.build_graph()
