# Ignore certain warning messages (such as Keras internal warnings)
import warnings
# Import commonly used modules from Keras
from keras import layers
from keras import models
from keras import initializers

# Define functions that build networks
# Parameters: imag_w, image_h: Width and height of the image ; class_num: Number of classes
def build_network(imag_w, image_h, class_num):
    # Ignore warning messages
    warnings.filterwarnings("ignore")

    # Initialize the sequence model
    model = models.Sequential()

    # BLOCK 1: Two convolutional layers + max pooling
    # First convolutional layer
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                            padding='same', name='block1_conv1', input_shape=(224, 224, 3)))
    # Second convolution layer
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                            padding='same', name='block1_conv2'))
    # Pooling layer, used to reduce feature maps
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))

    # BLOCK 2: Two convolutional layers + max pooling
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                            padding='same', name='block2_conv1'))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                            padding='same', name='block2_conv2'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool'))

    # BLOCK 3: Three convolutional layers + max pooling
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                            padding='same', name='block3_conv1'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                            padding='same', name='block3_conv2'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                            padding='same', name='block3_conv3'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool'))

    # BLOCK 4: Three convolutional layers + max pooling
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                            padding='same', name='block4_conv1'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                            padding='same', name='block4_conv2'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                            padding='same', name='block4_conv3'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool'))

    # BLOCK 5: Three convolutional layers + max pooling
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                            padding='same', name='block5_conv1'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                            padding='same', name='block5_conv2'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                            padding='same', name='block5_conv3'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool'))

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
    # Dropout again
    model.add(layers.Dropout(0.8))

    # Output layer, softmax activation, used for multi-classification tasks
    model.add(layers.Dense(class_num, activation='softmax'))

    # Print model structure
    model.summary()

    # Return the constructed model
    return model
