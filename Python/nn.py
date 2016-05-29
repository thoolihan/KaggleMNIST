import tensorflow as tf
import numpy as np
import pandas as pd
import time


def get_step_indexes(step, batch_size, n):
    start = (step * batch_size) % n
    stop = start + batch_size
    if stop > n:
        stop = n
    return (start, stop)

ratio = 0.7
batch_size = 1000
n_features = 784
label_index = 0
feature_index = range(0, n_features + 1)
feature_index.remove(label_index)
n_classes = 10
learning_rate = 0.001
keep_prob = 0.98
training_epochs = 2000
display_step = 100

x_all = np.loadtxt('../data/train.csv',
               delimiter = ',',
               skiprows = 1,
               dtype = np.float32,
               usecols = range(0, n_features + 1))

np.random.shuffle(x_all)

n_total = x_all.shape[0]
split = int(ratio * n_total)
n_hidden_1 = np.int32(np.floor(n_features / 2))
n_hidden_2 = np.int32(np.floor(n_features / 2))
n_hidden_3 = np.int32(np.floor(n_features / 2))

y_all = np.zeros((n_total, n_classes), np.float32)
digits = x_all[:, label_index]
for i, b in enumerate(digits):
    y_all[i, int(b)] = 1.0

x_train = x_all[:split, feature_index]
x_test = x_all[split:, feature_index]
y_train = y_all[:split, :]
y_test = y_all[split:, :]
n = x_train.shape[0]

x = tf.placeholder("float", [None, n_features])
y = tf.placeholder("float", [None, n_classes])
dropout = tf.placeholder("float")

def nn(_X, _weights, _biases, _dropout):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer_1 = tf.nn.dropout(layer_1, _dropout)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
    layer_2 = tf.nn.dropout(layer_2, _dropout)
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3']))
    layer_3 = tf.nn.dropout(layer_3, _dropout)
    return tf.matmul(layer_3, _weights['out']) + _biases['out']

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_uniform([n_features, n_hidden_1], -1.0, 1.0)),
    'h2': tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2], -1.0, 1.0)),
    'h3': tf.Variable(tf.random_uniform([n_hidden_2, n_hidden_3], -1.0, 1.0)),
    'out': tf.Variable(tf.random_uniform([n_hidden_2, n_classes], -1.0, 1.0))
}
biases = {
    'b1': tf.Variable(tf.random_uniform([n_hidden_1], -1.0, 1.0)),
    'b2': tf.Variable(tf.random_uniform([n_hidden_2], -1.0, 1.0)),
    'b3': tf.Variable(tf.random_uniform([n_hidden_3], -1.0, 1.0)),
    'out': tf.Variable(tf.random_uniform([n_classes], -1.0, 1.0))
}

# Construct model
pred = nn(x, weights, biases, dropout)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
tf.scalar_summary('cost', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

init = tf.initialize_all_variables()

with tf.Session() as sess:
    train_writer = tf.train.SummaryWriter('logs/', sess.graph)
    sess.run(init)

    batches = int(np.ceil(n / batch_size))
    merged = tf.merge_all_summaries()

    for epoch in xrange(training_epochs + 1):
        avg_cost = 0.0
        for step in range(batches):
            start, stop = get_step_indexes(step, batch_size, n)
            feed = {x: x_train[start:stop, :],
                    y: y_train[start:stop, :],
                    dropout: keep_prob }
            batch_cost, _ = sess.run([cost, optimizer], feed_dict = feed)
            avg_cost += batch_cost
        if epoch % display_step == 0 or epoch == training_epochs:
            summary = sess.run(merged, feed_dict = feed)
            train_writer.add_summary(summary, epoch)
            print 'epoch: %05d' % epoch, 'cost:', '{:.9f}'.format(avg_cost / batches)

    print("Optimization Complete")

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: x_test, y: y_test, dropout: 1.0})

    print("loading set for submission")
    x_test = np.loadtxt('../data/test.csv',
                        delimiter = ',',
                        skiprows = 1,
                        dtype = np.float32,
                        usecols = range(0, n_features))

    print("predicting against submission set")
    test_output = nn(x_test, weights, biases, dropout)
    output = sess.run(test_output, feed_dict = {x: x_test, dropout: keep_prob})

    labels = np.argmax(output, 1)
    image_ids = np.array(range(1, labels.shape[0] + 1))
    results = np.transpose(np.matrix([image_ids, labels]))

    print("writing submission file")
    df = pd.DataFrame(results, columns = ('ImageId', 'Label'))
    dstring = time.strftime('%Y.%m.%d.%H.%M.%S', time.gmtime())
    filename = 'output/nn.' + dstring + '.csv'
    df.to_csv(filename, index = False)
    print("wrote " + filename)
    print("done")
