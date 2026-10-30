# Import TensorFlow for the neural network and built-in MNIST dataset.
import tensorflow as tf  # Build, train, and save the model.
# Import matplotlib to display handwritten digit images.

import matplotlib.pyplot as plt  # Plot images and training progress.

# Import NumPy for choosing the most likely digit class.
import numpy as np  # Work with numeric arrays.

# Define the complete learning workflow in one function.
def main():

    # Load 60,000 training images and 10,000 test images from MNIST.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Convert pixel values from 0-255 into floating-point values from 0-1.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Build a simple dense neural network for the 10 digit classes.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape = (28, 28)),  # Each image is 28 by 28 pixels.
        tf.keras.layers.Flatten(),  # Turn each image into a 784-value vector.
        tf.keras.layers.Dense(128, activation = "relu"),  # Learn useful image patterns.
        tf.keras.layers.Dense(128, activation = "relu"),  # Learn a second set of patterns.
        tf.keras.layers.Dense(10, activation = "softmax")  # Return probabilities for digits 0-9.
    ])

    # Configure training with Adam and a multi-class classification loss.
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

    # Print a summary of the layers and parameter counts.
    model.summary()

    # Train using 10% of the training set for validation, not for final testing.
    model.fit(x_train, y_train, epochs = 5, batch_size = 32, validation_split = 0.1, verbose = 1)

    # Evaluate once on the separate test set.
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose = 0)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    # Predict one test image and choose the class with the highest probability.
    sample_index = 2
    probabilities = model.predict(x_test[sample_index:sample_index + 1], verbose = 0)
    predicted_digit = int(np.argmax(probabilities[0]))

    print(f"True digit: {int(y_test[sample_index])}")
    print(f"Predicted digit: {predicted_digit}")

    # Display the image with the predicted and true labels.
    plt.imshow(x_test[sample_index], cmap = "gray")
    plt.title(f"True: {y_test[sample_index]} | Predicted: {predicted_digit}")
    plt.axis("off")
    plt.show()

    # Save the trained model in Keras' native format for later use.
    model.save("mnist_digit_model.keras")
    print("Saved model: mnist_digit_model.keras")

# Start training when the script is run directly.
if __name__ == "__main__":
    main()
