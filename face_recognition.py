import os
import cv2 as cv
import numpy as np

def get_path_list(root_path):
    """
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory

        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    """
    path_list = os.listdir(root_path)

    return path_list

def get_class_id(root_path, train_names):
    """
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    """
    image_list = []
    image_classes_list = []
    for name in train_names:
        folder_path = root_path + "/" + name
        for image_path in os.listdir(folder_path):
            image = cv.imread(folder_path + "/" + image_path)
            image_list.append(image)
            image_classes_list.append(train_names.index(name) + 1)
        
    return (image_list, image_classes_list)

def detect_faces_and_filter(image_list, image_classes_list=None):
    """
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id

        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    """
    gray_face_list = []
    
    haarcascade_path = "model/haarcascade_frontalface_default.xml"
    detector = cv.CascadeClassifier(haarcascade_path)

    if image_classes_list:
        filtered_classes_list = []
        for image, image_class in zip(image_list, image_classes_list):
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            detected_faces = detector.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)

            if len(detected_faces) == 0:
                continue

            for bounding_box in detected_faces:
                x, y, w, h = bounding_box
                face_image = image[y:y + w, x:x + h]
                gray_face_list.append(face_image)
                filtered_classes_list.append(image_class)
        
        return (gray_face_list, None, filtered_classes_list)
    else:
        rectangles = []
        for image in image_list:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            detected_faces = detector.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)

            if len(detected_faces) == 0:
                continue

            for bounding_box in detected_faces:
                x, y, w, h = bounding_box
                face_image = image[y:y + w, x:x + h]
                gray_face_list.append(face_image)
                rectangles.append(bounding_box)

        return (gray_face_list, rectangles, None)

def train(train_face_grays, image_classes_list):
    """
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id

        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    """
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays, np.array(image_classes_list))

    return recognizer

def get_test_images_data(test_root_path):
    """
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory

        Returns
        -------
        list
            List containing all loaded test images
    """
    test_image_list = []
    for path in os.listdir(test_root_path):
        image = cv.imread(test_root_path + "/" + path)
        test_image_list.append(image)

    return test_image_list

def predict(recognizer, test_faces_gray):
    """
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    """
    predict_results = []
    for image in test_faces_gray:
        predict_results.append(recognizer.predict(image))

    return predict_results

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    """
        To draw prediction results on the given test images and acceptance status

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    """
    predicted_test_image_list = []
    for index, (image, rectangle) in enumerate(zip(test_image_list, test_faces_rects)):
        x, y, h, w = rectangle

        if train_names[index] == "Jokowi":
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 10)
            cv.putText(image, train_names[index] + " (Active)", (x - 75, y + w + 75), cv.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 5)
        else:
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 10)
            cv.putText(image, train_names[index] + " (Inactive)", (x - 75, y + w + 75), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5)

        predicted_test_image_list.append(image)

    return predicted_test_image_list

def combine_and_show_result(image_list):
    """
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    """
    images = []
    for image in image_list:
        width = int(image.shape[1] * 0.6)
        height = int(image.shape[0] * 0.6)
        images.append(cv.resize(image, (width, height), interpolation=cv.INTER_AREA))

    upper_inactive = cv.hconcat((images[0], images[1]))
    lower_inactive = cv.hconcat((images[3], images[4]))
    inactive = cv.vconcat((upper_inactive, lower_inactive))
    cv.imshow("Indonesian Presidents", cv.hconcat((inactive, images[2])))
    cv.waitKey(0)


if __name__ == '__main__':

    train_root_path = "data/train"
    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    test_root_path = "data/test"
    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)

    combine_and_show_result(predicted_test_image_list)
