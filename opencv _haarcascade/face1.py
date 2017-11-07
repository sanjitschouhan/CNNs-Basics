#!/usr/bin/python

# Import the required modules
import os
import cv2
import numpy as np
from PIL import Image

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()


def get_images_and_labels(path, dataset):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    path = os.path.join(path, dataset)
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    labels_dic = {}
    ind = 0
    for person in os.listdir(path):
        ind += 1
        labels_dic[ind] = person
        image_paths = [os.path.join(path, person, f)
                       for f in os.listdir(os.path.join(path, person))]
        for image in image_paths:
            img = Image.open(image).convert('L')
            img = np.array(img, 'uint8')
            faces = faceCascade.detectMultiScale(img)
            for (x, y, w, h) in faces:
                labels.append(ind)
                images.append(img[y: y + h, x: x + w])
    return images, labels, labels_dic


# Path to the Yale Dataset
path = './faces'
# Call the get_images_and_labels function and get the face images and the
# corresponding labels
train_images, train_labels, labels_dic = get_images_and_labels(path, 'train')
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(train_images, np.array(train_labels))

test_image, test_labels, test_dic = get_images_and_labels(path, 'test')
# Append the images with the extension .sad into image_paths
correct_classification = 0
for i in range(len(test_image)):
    nbr_predicted, conf = recognizer.predict(test_image[i])
    nbr_predicted = labels_dic[nbr_predicted]
    nbr_actual = test_dic[test_labels[i]]
    if nbr_actual == nbr_predicted:
        print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
        correct_classification += 1
    else:
        print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
    cv2.imshow("Recognizing Face", test_image[i])
    cv2.waitKey(100)

print "\n\nTotal Test Images:", len(test_image)
print "Correctly Classified:", correct_classification
print "Accuracy:", correct_classification * 100.0 / len(test_image)
