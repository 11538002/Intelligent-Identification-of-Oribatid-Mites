# Ignore certain warning messages (such as Keras internal warnings)
import warnings
# Import layer module
from keras import layers
# Import model module
from keras import models
# Import initializer module
from keras import initializers

# Define functions that build networks
# Parameters: imag_w, image_h: Width and height of the image ; class_num: Number of classes
def build_network(imag_w, image_h, class_num):
    # Ignore warning messages
    warnings.filterwarnings("ignore")

    # Using the Sequential model
    model = models.Sequential()

    # First convolutional layer: 11x11 convolution kernel, stride 4 (simulating the initial layer of AlexNet)
    model.add(layers.Conv2D(48, (11, 11),
                            # Input the image size and number of channels (3 indicates an RGB image)
                            input_shape=(imag_w, image_h, 3),
                            activation='relu',
                            padding="same",
                            # Step size 4, reduce feature map size
                            strides=(4, 4),
                            kernel_initializer=initializers.initializers_v1.TruncatedNormal(stddev=0.01)
                            ))
    # Batch normalization improves training stability
    model.add(layers.BatchNormalization())
    # Max pooling, further downsampling
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Second convolutional layer: 5x5 convolution kernel, stride 1
    model.add(layers.Conv2D(128, (5, 5),
                            activation='relu',
                            padding="same",
                            strides=(1, 1),
                            kernel_initializer=initializers.initializers_v2.TruncatedNormal(stddev=0.01)
                            ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Third convolution layer: 3x3 convolution kernel
    model.add(layers.Conv2D(384, (3, 3),
                            activation='relu',
                            padding="same",
                            strides=(1, 1),
                            kernel_initializer=initializers.initializers_v2.TruncatedNormal(stddev=0.01)
                            ))
    model.add(layers.BatchNormalization())

    # Fourth convolution layer: Continue using the 3x3 convolution kernel.
    model.add(layers.Conv2D(384, (3, 3),
                            activation='relu',
                            padding="same",
                            strides=(1, 1),
                            kernel_initializer=initializers.initializers_v2.TruncatedNormal(stddev=0.01)
                            ))
    model.add(layers.BatchNormalization())

    # 5th convolution layer: output channels reduced to 256
    model.add(layers.Conv2D(256, (3, 3),
                            activation='relu',
                            padding="same",
                            strides=(1, 1),
                            kernel_initializer=initializers.initializers_v2.TruncatedNormal(stddev=0.01)
                            ))
    model.add(layers.BatchNormalization())
    # Last pooling
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Flatten the multidimensional output into one dimension to prepare for the fully connected layer
    model.add(layers.Flatten())

    # First full-connected layer: 4096 neurons, initialized using TruncatedNormal
    model.add(layers.Dense(4096, activation='relu',
                           kernel_initializer=initializers.initializers_v2.TruncatedNormal(stddev=0.01)))
    # Discard 80% of neurons to prevent overfitting
    model.add(layers.Dropout(0.8))

    # Second full-connected layer
    model.add(layers.Dense(4096, activation='relu',
                           kernel_initializer=initializers.initializers_v2.TruncatedNormal(stddev=0.01)))
    model.add(layers.Dropout(0.8))

    # Output layer, softmax activation, used for multi-classification tasks
    model.add(layers.Dense(class_num, activation='softmax'))

    # Print model structure
    model.summary()

    # Return the constructed model
    return model
