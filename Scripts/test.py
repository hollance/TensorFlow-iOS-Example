# This script tests how well the trained model performs on the portion of the 
# data that was not used for training.

import os
import numpy as np
import tensorflow as tf
from sklearn import metrics

checkpoint_dir = "/tmp/voice/"

# Load the test data.
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Test set size:", X_test.shape)

with tf.Session() as sess:
    # Load the graph.
    graph_file = os.path.join(checkpoint_dir, "graph.pb")
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

    # Uncomment the next line in case you're curious what the graph looks like.
    #print(graph_def.ListFields())

    # Get the model's variables.
    W = sess.graph.get_tensor_by_name("model/W:0")
    b = sess.graph.get_tensor_by_name("model/b:0")

    # Load the saved variables from the checkpoint back into the session.
    checkpoint_file = os.path.join(checkpoint_dir, "model")
    saver = tf.train.Saver([W, b])
    saver.restore(sess, checkpoint_file)

    # Get the placeholders and the accuracy operation, so that we can compute
    # the accuracy (% correct) of the test set.
    x = sess.graph.get_tensor_by_name("inputs/x-input:0")
    y = sess.graph.get_tensor_by_name("inputs/y-input:0")
    accuracy = sess.graph.get_tensor_by_name("score/accuracy:0")
    print("Test set accuracy:", sess.run(accuracy, feed_dict={x: X_test, y: y_test}))

    # Also show some other reports.
    inference = sess.graph.get_tensor_by_name("inference/inference:0")
    predictions = sess.run(inference, feed_dict={x: X_test})
    print("\nClassification report:")
    print(metrics.classification_report(y_test.ravel(), predictions))
    print("Confusion matrix:")
    print(metrics.confusion_matrix(y_test.ravel(), predictions))
