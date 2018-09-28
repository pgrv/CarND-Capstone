from styx_msgs.msg import TrafficLight
from keras.models import load_model
import numpy as np
import tensorflow as tf

class TLClassifier(object):
    def __init__(self, is_site):
        #TODO load classifier
        if is_site:
            self.classifier = load_model('light_classification/real_classifier.h5')
        else:
            self.classifier = load_model('light_classification/sim_classifier.h5')
        self.classifier._make_predict_function()
        self.graph = tf.get_default_graph()
        #pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        img_resized = np.resize(image, (1, 600, 800, 3))
        with self.graph.as_default():
            tlClass = self.classifier.predict(img_resized)
            if(np.argmax(tlClass) == 0):
                return TrafficLight.RED
            elif(np.argmax(tlClass) == 1):
                return TrafficLight.GREEN
            elif(np.argmax(tlClass) == 2):
                return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN
