# This script exports the learned parameters so that we can use them from Metal.

# Note: Dor this simple demo project the weight matrix is only 20 values and the bias
# is a single number. With such a simple model you might as well stick the parameters
# inside a static array in the iOS app source code. In practice, however, most models 
# will have millions of parameters.

import os
import numpy as np
import tensorflow as tf
from sklearn import metrics

checkpoint_dir = "/tmp/voice/"

with tf.Session() as sess:
    # Load the graph.
    graph_file = os.path.join(checkpoint_dir, "graph.pb")
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

    # Get the model's variables.
    W = sess.graph.get_tensor_by_name("model/W:0")
    b = sess.graph.get_tensor_by_name("model/b:0")

    # Load the saved variables from the checkpoint back into the session.
    checkpoint_file = os.path.join(checkpoint_dir, "model")
    saver = tf.train.Saver([W, b])
    saver.restore(sess, checkpoint_file)

    # Just for debugging, print out the learned parameters.
    print("W:", W.eval())
    print("b:", b.eval())
    
    # Export the contents of W and b as binary files.
    W.eval().tofile("W.bin")
    b.eval().tofile("b.bin")
