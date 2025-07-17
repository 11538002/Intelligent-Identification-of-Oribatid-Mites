# Ignore certain warning messages (such as Keras internal warnings)
import warnings
# Import ResNet series model builders from Keras models
from keras.applications.resnet import ResNet, ResNet101, ResNet50
# Import commonly used layer modules
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
# Import model constructor
from keras.models import Model

# Define model construction function
# Parameters: imag_w, image_h: Width and height of the image ; class_num: Number of classes
def build_network(imag_w, image_h, class_num):
    # Ignore Keras warning messages
    warnings.filterwarnings("ignore")

    # Build a ResNet model without the top full-connected classification layer
    base_model = ResNet101(
        # Do not use the original top full-connected classification layer
        include_top=False,
        # Use weights pre-trained on ImageNet
        weights='imagenet',
        # Input the image size and number of channels (3 indicates an RGB image)
        input_shape=(imag_w, image_h, 3)
    )

    # Obtain the output feature map of ResNet
    x = base_model.output

    # Add a global average pooling layer to compress each channel into a single value
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
