import requests

import keys

FACE_API_KEY = keys.FACE_API_KEY
FACE_API_ENDPOINT = keys.FACE_API_ENDPOINT


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

    def send_request_to_azure(self, image):
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
            response = self.send_request_to_azure(image)
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
