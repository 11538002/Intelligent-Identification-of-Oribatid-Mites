# The environment configured for this experiment is Python 3.9, Keras 2.6, and TensorFlow 2.6.
# Import the required libraries
import numpy as np
import tensorflow as tf
# Used for image processing
import cv2
# Used for file processing and progress bar display
import os, glob, tqdm
# Used for dataset partitioning
from sklearn.model_selection import train_test_split
# optimizer
from keras.optimizer_v2 import adam
from tensorflow.keras.utils import to_categorical

# Import customized network structure modules: for comparison experiments
import train_densenet
import train_resnet
import train_alexnet
import train_vgg16

# Initialize image list and label list
img_list, label_list = [], []

# Read all subdirectory names in the folder where the image is located
labels = os.listdir("D:\\oribatid\\Raw images(use microscope)")
# Print categories
print(labels)

# Iterate through each category
for label in labels:
    # Get the paths of all .jpg images in the current category folder
    file_list = glob.glob(f'D:\\oribatid\\Raw images(use microscope)\\{label}/*.jpg')

    # Process all images in this category
    for img_file in tqdm.tqdm(file_list, desc=f"处理{label}"):
        # Read image
        img = cv2.imread(img_file)
        # Resize the image to 224x224: firstly, because DenseNet requires the input image size to be 224x224, and secondly, to improve the computational efficiency of the program
        img_resize = cv2.resize(img, (224, 224))
        # Add the transformed image to the dataset
        img_list.append(img_resize)
        label_list.append(label)

        # Data augmentation 1: Replacing the background color of images for preprocessing and comparison experiments
        information = img_resize.shape
        for row in range(information[0]):
            for col in range(information[1]):
                (b, g, r) = img_resize[row, col]
                if b >= 150:
                    img_resize[row, col] = (255, 255, 255)

        # Data augmentation 2: Randomly select a method to transform the aspect ratio and fill in the image so that the image size remains 224×224
        index = np.random.randint(1, 5)
        # Change the image's length and width to 4:5
        if index == 1:
            img_resize = cv2.resize(img_resize, None, fx=0.5, fy=0.625)
            img_resize = cv2.copyMakeBorder(img_resize, 42, 42, 56, 56, borderType=cv2.BORDER_CONSTANT)
        # Change the image's length and width to 7:8
        elif index == 2:
            img_resize = cv2.resize(img_resize, None, fx=1, fy=0.875)
            img_resize = cv2.copyMakeBorder(img_resize, 14, 14, 0, 0, borderType=cv2.BORDER_CONSTANT)
        # Change the image's length and width to 3:2
        elif index == 3:
            img_resize = cv2.resize(img_resize, None, fx=0.75, fy=0.5)
            img_resize = cv2.copyMakeBorder(img_resize, 56, 56, 28, 28, borderType=cv2.BORDER_CONSTANT)
        # When index == 4, no size conversion is performed, and the ratio remains 1:1

        # Add the transformed image to the dataset.
        img_list.append(img_resize)
        label_list.append(label)

        # Data augmentation 3: Randomly flip images (-1: horizontal + vertical, 0: vertical, 1: horizontal)
        flip_direction = np.random.randint(-1, 2)
        flipped_image = cv2.flip(img_resize, flip_direction)
        # Add the transformed image to the dataset.
        img_list.append(flipped_image)
        label_list.append(label)

        # Data augmentation 4: Randomly rotate images (angle range: 30° to 330°)
        angle = np.random.randint(30, 330)
        rows, cols = img_resize.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(img_resize, M, (cols, rows))
        # Add the transformed image to the dataset.
        img_list.append(rotated_image)
        label_list.append(label)

        # Data augmentation 5: Adding Gaussian noise
        mean = 0
        var = 0.5
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, img_resize.shape).astype('uint8')
        noisy_image = cv2.add(img_resize, gaussian)
        # Add the transformed image to the dataset.
        img_list.append(noisy_image)
        label_list.append(label)

        # Data augmentation 6: HSV color space transformation, hue channel offset 30
        hsv_image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + 30) % 180
        result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        # Add the transformed image to the dataset.
        img_list.append(result_image)
        label_list.append(label)

# Convert images and labels to numpy array format
X = np.array(img_list)
y = np.array(label_list)

# Save preprocessed images and labels
np.savez('img_data.npz', X, y)

# Read saved data
data = np.load('img_data.npz')
img_list = data["arr_0"]
label_list = data["arr_1"]

# Convert string type labels to numeric encoding
# Get unique label
label_names = np.unique(label_list)
# Map labels to indexes
labels_to_index = dict((name, i) for i, name in enumerate(label_names))
# Convert to a list of digital labels
all_labels = [labels_to_index.get(name) for name in label_list]

# Randomize image and label order
np.random.seed(0)
random_index = np.random.permutation(len(img_list))
img_list = np.array(img_list)[random_index]
all_labels = np.array(all_labels)[random_index]

# Divide the dataset into a 75% training / 25% testing ratio
x_train, x_test, y_train, y_test = train_test_split(img_list, all_labels, test_size=0.25, shuffle=True)

# Output dimension information for training and testing sets
print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test))

# Call a custom network construction model
model = train_densenet.build_network(224, 224, len(labels))

# Compile the model using the cross-entropy loss function, Adam optimizer, and accuracy metric
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=adam.Adam(0.00001),
              metrics=['acc'])

# Model training, set the number of training iterations to 15 and the batch size to 20.
model.fit(np.asarray(x_train), to_categorical(y_train), epochs=15, batch_size=20)

# Save model to specified path
model.save('exp_tiaochong/model_res')

# Evaluate model performance on the testing set
loss, accuracy = model.evaluate(np.asarray(x_test), to_categorical(y_test), batch_size=20, verbose=1)
# Output testing set loss
print('Test loss:', loss)
# Output testing set accuracy
print('Test accuracy:', accuracy)

# Use the test set for prediction (same effect as model.evaluate; both are methods of evaluating the model using the test set).
predict = model.predict(x_test)
# Obtain the category corresponding to the maximum probability of the prediction
pre_labels = np.argmax(predict, axis=1)

# Compare predicted labels and actual labels, and calculate accuracy
correct_label = np.equal(y_test, pre_labels)
accuracy = np.mean(correct_label)
# Printing prediction accuracy
print("pre acc=", accuracy)
