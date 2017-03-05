# This script is used to train the model. It repeats indefinitely and saves the
# model every so often to a checkpoint. 
#
# Press Ctrl+C when you feel that training has gone on long enough (since this is 
# only a simple model it takes less than a minute to train, but a training a deep l
# earning model could take days).

import os
import numpy as np
import tensorflow as tf

checkpoint_dir = "/tmp/voice/"
print_every = 1000
save_every = 10000
num_inputs = 20
num_classes = 1

# Load the training data.
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

print("Training set size:", X_train.shape)

# Below we'll define the computational graph using TensorFlow. The different parts 
# of the model are grouped into different "scopes", making it easier to understand
# what each part is doing.

# Hyperparameters let you configure the model and how it is trained. They're
# called "hyper" parameters because unlike the regular parameters they are not
# learned by the model -- you have to set them to appropriate values yourself.
#
# The learning_rate tells the optimizer how big of a steps it should take.
# Regularization is used to prevent overfitting on the training set.
with tf.name_scope("hyperparameters"):
    regularization = tf.placeholder(tf.float32, name="regularization")
    learning_rate = tf.placeholder(tf.float32, name="learning-rate")

# This is where we feed the training data (and later the test data) into the model. 
# In this dataset there are 20 features, so x is a matrix with 20 columns. Its number 
# of rows is None because it depends on how many examples at a time we put into this 
# matrix. This is a binary classifier so for every training example, y gives a single 
# output: 1 = male, 0 = female.
with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float32, [None, num_inputs], name="x-input")
    y = tf.placeholder(tf.float32, [None, num_classes], name="y-input")
    
# The parameters that we'll learn consist of W, a weight matrix, and b, a vector
# of bias values. (Actually, b is just a single value since the classifier has only
# one output. For a classifier that can recognize multiple classes, b would have as
# many elements as there are classes.)
with tf.name_scope("model"):
    W = tf.Variable(tf.zeros([num_inputs, num_classes]), name="W")
    b = tf.Variable(tf.zeros([num_classes]), name="b")

    # The output is the probability the speaker is male. If this is greater than
    # 0.5, we consider the speaker to be male, otherwise female.
    y_pred = tf.sigmoid(tf.matmul(x, W) + b, name="y_pred")

# This is a logistic classifier, so the loss function is the logistic loss.
with tf.name_scope("loss-function"):
    loss = tf.losses.log_loss(labels=y, predictions=y_pred)
    
    # Add L2 regularization to the loss.
    loss += regularization * tf.nn.l2_loss(W)

# Use the ADAM optimizer to minimize the loss.
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

# For doing inference on new data for which we don't have labels.
with tf.name_scope("inference"):
    inference = tf.to_float(y_pred > 0.5, name="inference")

# The accuracy operation computes the % correct on a dataset with known labels. 
with tf.name_scope("score"):
    correct_prediction = tf.equal(inference, y)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction), name="accuracy")

init = tf.global_variables_initializer()

# For writing training checkpoints and reading them back in.
saver = tf.train.Saver()
tf.gfile.MakeDirs(checkpoint_dir)

with tf.Session() as sess:
    # Write the graph definition to a file. We'll load this in the test.py script.
    tf.train.write_graph(sess.graph_def, checkpoint_dir, "graph.pb", False)

    # Reset W and b to zero.
    sess.run(init)

    # Sanity check: the initial loss should be 0.693146, which is -ln(0.5).
    loss_value = sess.run(loss, feed_dict={x: X_train, y: y_train, regularization: 0})
    print("Initial loss:", loss_value)

    # Loop forever:
    step = 0
    while True:
        # We randomly shuffle the examples every time we train.
        perm = np.arange(len(X_train))
        np.random.shuffle(perm)
        X_train = X_train[perm]
        y_train = y_train[perm]

        # Run the optimizer over the entire training set at once. For larger datasets
        # you would train in batches of 100-1000 examples instead of the entire thing.
        feed = {x: X_train, y: y_train, learning_rate: 1e-2, regularization: 1e-5}
        sess.run(train_op, feed_dict=feed)

        # Print the loss once every so many steps. Because of the regularization, 
        # at some point the loss won't become smaller anymore. At that point, it's
        # safe to press Ctrl+C to stop the training.
        if step % print_every == 0:
            train_accuracy, loss_value = sess.run([accuracy, loss], feed_dict=feed)
            print("step: %4d, loss: %.4f, training accuracy: %.4f" % \
                    (step, loss_value, train_accuracy))

        step += 1

        # Save the model. You should only press Ctrl+C after you see this message.
        if step % save_every == 0:
            checkpoint_file = os.path.join(checkpoint_dir, "model")
            saver.save(sess, checkpoint_file)            
            print("*** SAVED MODEL ***")
