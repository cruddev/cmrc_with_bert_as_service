import tensorflow as tf
from bert_serving.client import BertClient
import pandas as pd

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
df = pd.read_csv("D:/data/QATest/dev.csv", sep=";")
bc = BertClient(ip='localhost')
batch_size = 2
hidden_size = 768
seq_length = 256
init_learning_rate = 3e-5
num_train_step = 1000
def get_bert_embedding(question_seq_content):
    input_embedding = bc.encode(question_seq_content)
    return input_embedding

batch_sample = [df["question"][0] + " ||| " + df["content"][0], df["question"][1] + " ||| " + df["content"][1]]
batch_bert_embedding = get_bert_embedding(batch_sample)
output_weights = tf.get_variable("output_weights", [2, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
output_bias = tf.get_variable("output_bias", [2], initializer=tf.zeros_initializer())
hidden_matrix = tf.reshape(batch_bert_embedding, [batch_size * seq_length, hidden_size])
logits = tf.matmul(hidden_matrix, output_weights, transpose_b=True)
logits = tf.nn.bias_add(logits, output_bias)
logits = tf.reshape(logits, [batch_size, seq_length, 2])
logits = tf.transpose(logits, [2, 0, 1])
unstacked_logits = tf.unstack(logits, axis=0)
start_logits = unstacked_logits[0]
end_logits = unstacked_logits[1]

def compute_loss(logits, positions):
    one_hot_positions = tf.one_hot(positions, depth=seq_length, dtype=tf.float32)
    log_probs = tf.nn.softmax(logits, axis=-1)
    loss = -tf.reduce_mean(tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
    return loss

global_step = tf.train.get_or_create_global_step()
start_loss = compute_loss(start_logits, batch_start_positions)
end_loss = compute_loss(end_logits, batch_end_positions)
total_loss = (start_loss + end_loss) / 2.0
learning_rate = tf.constant(value=init_learning_rate, shape=[], dtype=tf.float32)
learning_rate = tf.train.polynomial_decay(learning_rate, global_step, num_train_step, end_learning_rate=0.0, power=1.0, cycle=False)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())

for i in range(num_train_step):
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[0:1000]))
