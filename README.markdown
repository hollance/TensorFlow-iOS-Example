# TensorFlow on iOS demo

This is the code that accompanies my blog post [Getting started with TensorFlow on iOS](http://machinethink.net/blog/tensorflow-on-ios/).

It uses TensorFlow to train a basic binary classifier on the [Gender Recognition by Voice and Speech Analysis](https://www.kaggle.com/primaryobjects/voicegender) dataset.

This project includes the following:

- The dataset in the file **voice.csv**.
- Python scripts to train the model with TensorFlow on your Mac.
- An iOS app that uses the TensorFlow C++ API to do inference.
- An iOS app that uses Metal to do inference using the trained model.

## Training the model

To train the model, do the following:

1. Make sure these are installed: `python3`, `numpy`, `pandas`, `scikit-learn`, `tensorflow`.
2. Run the **split_data.py** script to divide the dataset into a training set and a test set. This creates 4 new files: `X_train.npy`, `y_train.npy`, `X_test.npy`, and `y_test.npy`.
3. Run the **train.py** script. This trains the logistic classifier and saves the model to `/tmp/voice` every 10,000 training steps. Training happens in an infinite loop and goes on forever, so press Ctrl+C when you're happy with the training set accuracy and the loss no longer becomes any lower.
4. Run the **test.py** script to compute the accuracy on the test set. This also prints out a report with precision / recall / f1-score and a confusion matrix.

## Using the model with the iOS TensorFlow app

To run the model on the iOS TensorFlow app, do the following:

1. Clone [TensorFlow](https://github.com/tensorflow) and [build the iOS library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile).
2. Open the **VoiceTensorFlow** Xcode project. In **Build Settings**, **Other Linker Flags** and **Header Search Paths**, change the paths to your local installation of TensorFlow.

The model is already included in the app as **inference.pb**. If you train the model with different settings, you need to run the `freeze_graph` and `optimize_for_inference` tools to create a new inference.pb.

## Using the model with the iOS Metal app

To run the model on the iOS Metal app, do the following:

1. Run the **export_weights.py** script. This creates two new files that contain the model's learned parameters: `W.bin` for the weights and `b.bin` for the bias.
2. Copy `W.bin` and `b.bin` into the **VoiceMetal** Xcode project and build the app.

You need to run the Metal app on a device, it won't work in the simulator.

