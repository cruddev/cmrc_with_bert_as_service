import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from bert_serving.client import BertClient
import DataReader as data_reader
import tensorflow as tf
import random
import Predictor as predictor
# bert-serving-start -model_dir D:/model/multi_cased_L-12_H-768_A-12/ -max_seq_len 128 -pooling_strategy NONE -show_tokens_to_client -cased_tokenization
from bert_serving.client import BertClient
bc = BertClient(ip='localhost')
train_file = 'D:/data/cmrc_squad/cmrc2018_trial.json'
test_file = 'D:/data/cmrc_squad/cmrc2018_trial.json'
max_query_length = 64
max_seq_length = 128
doc_stride = 128
batch_size = 16
hidden_size = 768
num_epoch = 2;
# 0.0001
init_lr = 3e-2
tf.logging.set_verbosity(tf.logging.INFO)

train_data = data_reader.read_squad_examples(train_file, True)
test_data = data_reader.read_squad_examples(test_file, True)
test_data = test_data[0:4]
train_data_collector = []
test_data_collector = []
data_reader.convert_examples_to_features(train_data, max_query_length=max_query_length, max_seq_length=max_seq_length,
                                         doc_stride=doc_stride, is_training=True, data_collector=train_data_collector,
                                         bert_client=bc)
data_reader.convert_examples_to_features(test_data, max_query_length=max_query_length, max_seq_length=max_seq_length,
                                         doc_stride=doc_stride, is_training=True, data_collector=test_data_collector,
                                         bert_client=bc)

num_train_steps = len(train_data_collector) / batch_size

# 定义计算图
start_position = tf.placeholder(shape=[None], dtype=tf.int32)
end_position = tf.placeholder(shape=[None], dtype=tf.int32)
input_embedding = tf.placeholder(shape=[None, None, None], dtype=tf.float32)
output_weights = tf.get_variable(
    "output_weights", [2, hidden_size],
    initializer=tf.truncated_normal_initializer(stddev=0.02))

output_bias = tf.get_variable(
    "output_bias", [2], initializer=tf.zeros_initializer())
seq_length = max_seq_length
final_hidden_matrix = tf.reshape(input_embedding,
                                 [batch_size * seq_length, hidden_size])
logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
logits = tf.nn.bias_add(logits, output_bias)

logits = tf.reshape(logits, [batch_size, seq_length, 2])
logits = tf.transpose(logits, [2, 0, 1])

unstacked_logits = tf.unstack(logits, axis=0)

start_logits = unstacked_logits[0]
end_logits = unstacked_logits[1]

def compute_loss(logits, positions):
    one_hot_positions = tf.one_hot(
        positions, depth=seq_length, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    loss = -tf.reduce_mean(
        tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
    return loss

start_loss = compute_loss(start_logits, start_position)
end_loss = compute_loss(end_logits, end_position)
loss = (2 * start_loss + end_loss) / 2.0
# train_op = tf.train.AdamOptimizer(init_lr).minimize(loss)
train_op = tf.train.GradientDescentOptimizer(init_lr).minimize(loss)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
# learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

# learning_rate = tf.train.polynomial_decay(
#     learning_rate,
#     global_step,
#     num_train_steps,
#     end_learning_rate=0.0,
#     power=1.0,
#     cycle=False)

class Result(object):
    def __init__(self,
                 unique_id,
                 start_logits=None,
                 end_logits=None):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits

for epoch in range(0, num_epoch):
    random.shuffle(train_data_collector)
    mini_batches = [
        train_data_collector[k:k + batch_size]
        for k in range(0, len(train_data_collector), batch_size)]
    for mini_batch in mini_batches:
        train_tokens = []
        train_start_index = []
        train_end_index = []
        for feature in mini_batch:
            train_tokens.append(feature.tokens[1:-1])
            train_start_index.append(feature.start_position)
            train_end_index.append(feature.end_position)
        _, loss_value = sess.run([train_op, loss], feed_dict={input_embedding: bc.encode(train_tokens, is_tokenized=True),
                                                              start_position: train_start_index, end_position:train_end_index})
        tf.logging.info("loss value: %s" % str(loss_value))
        test_tokens = []
        for feature in test_data_collector:
            test_tokens.append(feature.tokens[1:-1])
        predicted_logits = sess.run(unstacked_logits, feed_dict={input_embedding: bc.encode(test_tokens, is_tokenized=True)})
        all_results = []
        start_logits = predicted_logits[0]
        end_logits = predicted_logits[1]
        for result_index in range(0, len(test_data_collector)):
            result = Result(
                unique_id = test_data_collector[result_index].unique_id,
                start_logits = sess.run(tf.nn.softmax(start_logits[result_index])),
                end_logits = sess.run(tf.nn.softmax(end_logits[result_index])))
            all_results.append(result)
        predictor.write_predictions(all_examples=test_data, all_features=test_data_collector, all_results=all_results,
                                    max_answer_length=30, n_best_size=5)

