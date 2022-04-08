import keras
import requests
from keras.preprocessing.image import img_to_array
import keys
import cv2

FACE_API_KEY = keys.FACE_API_KEY
FACE_API_ENDPOINT = keys.FACE_API_ENDPOINT
FACE_API_ENDPOINT_VERIFY = keys.FACE_API_ENDPOINT_VERIFY


class TooManyFacesException(Exception):
    def __init__(self):
        self.message = "Too many faces exception"
        super().__init__(self.message)


class NoFaceException(Exception):
    def __init__(self):
        self.message = "No face exception"
        super().__init__(self.message)


class ServerNotRespondingException(Exception):
    def __init__(self):
        self.message = "Problem with Azure's endpoint"
        super().__init__(self.message)


class BlurredFaceException(Exception):
    def __init__(self):
        self.message = "Face in the photo is blurred"
        super().__init__(self.message)


def check_if_face_frontal(response_json):
    roll = int(response_json[0]['faceAttributes']['headPose']['roll'])
    if abs(roll) > 2.5:
        return False
    return True


def check_if_face_blurred(response_json):
    blur = str(response_json[0]['faceAttributes']['blur']['blurLevel'])
    if blur == "high":
        return True
    return False


class AzureRequestSender:

    def __init__(self, params):
        self.params = params

    def send_image_request_to_azure(self, image):
        headers = {
            'Content-Type': 'application/octet-stream',
            'Ocp-Apim-Subscription-Key': FACE_API_KEY,
        }

        return requests.post(FACE_API_ENDPOINT,
                             params=self.params,
                             headers=headers,
                             data=image.read())

    def detect_face(self, image):
        try:
            response = self.send_image_request_to_azure(image)
        except KeyError:
            raise ServerNotRespondingException
        else:
            response_json = response.json()
            if len(response_json) > 1:
                raise TooManyFacesException
            if len(response_json) == 0:
                raise NoFaceException
            if check_if_face_blurred(response_json):
                raise BlurredFaceException
            return response_json[0]

    @staticmethod
    def compare_faces(face1_id, face2_id):
        headers = {
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': FACE_API_KEY,
        }
        body = {"faceId1": face1_id, "faceId2": face2_id}

        params = {}
        response = requests.post(FACE_API_ENDPOINT_VERIFY,
                                 params=params,
                                 headers=headers,
                                 json=body)
        return response.json()


class KerasEmotionModelConnector:

    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def detect_emotion(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_for_pred = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_CUBIC)
        img_for_pred = img_for_pred.reshape(-1, 48, 48)
        prediction = self.model.predict([img_for_pred])
        return prediction

