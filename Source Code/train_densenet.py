# Ignore certain warning messages (such as Keras internal warnings)
import warnings
# Import the layers needed to construct a neural network
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
# Import model constructors
from keras.models import Model
# Import DenseNet
from keras.applications.densenet import DenseNet

# Define functions that build networks
# Parameters: imag_w, image_h: Width and height of the image ; class_num: Number of classes
def build_network(imag_w, image_h, class_num):
    # Ignore warning messages
    warnings.filterwarnings("ignore")

    # Load the DenseNet model as the base network, excluding the top full-connected layer
    base_model = DenseNet(
        # Configure the number of layers for each dense block
        blocks=[6, 12, 48, 32],
        # Does not include the final full-connected layer
        include_top=False,
        # Use weights pre-trained on ImageNet
        weights='imagenet',
        # Input the image size and number of channels (3 indicates an RGB image)
        input_shape=(imag_w, image_h, 3)
    )

    # Obtain the output tensor of base_model
    x = base_model.output

    # Add a global average pooling layer to reduce the feature map to one value for each channel
    x = GlobalAveragePooling2D()(x)

    # Add a Dropout layer to prevent overfitting (discard 50% of neurons)
    x = Dropout(0.5)(x)

    # Add a full-connected layer using the ReLU activation function with an output dimension of 1024
    x = Dense(1024, activation='relu')(x)

    # Output layer, using the softmax activation function, corresponding to the number of classifications
    predictions = Dense(class_num, activation='softmax')(x)

    # Build the final model, with the input being the input of base_model and the output being our newly added predictions layer
    model = Model(inputs=base_model.input, outputs=predictions)

    # Print model structure
    model.summary()

    # Return the constructed model
    return model
