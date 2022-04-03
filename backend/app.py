from flask import Flask, jsonify, request

from detection import AzureRequestSender

EMOTION_PARAMS = {
    'returnFaceId': 'false',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'headPose,blur,emotion'
}

KEYPOINTS_PARAMS = {
    'returnFaceId': 'false',
    'returnFaceLandmarks': 'true',
    'returnFaceAttributes': 'headPose,blur'
}

COMPARISON_PARAMS = {'returnFaceId': 'true',
                     'returnFaceLandmarks': 'false',
                     'returnFaceAttributes': 'headPose,blur'}

EMOTIONS_DETECTION_MODEL = AzureRequestSender(params=EMOTION_PARAMS)
KEYPOINTS_DETECTION_MODEL = AzureRequestSender(params=KEYPOINTS_PARAMS)
COMPARISON_DETECTION_MODEL = AzureRequestSender(params=COMPARISON_PARAMS)

app = Flask(__name__)
app.debug = True



def detect_keypoints(img):
    res = KEYPOINTS_DETECTION_MODEL.detect_face(img)
    return res


def detect_id(img):
    res = COMPARISON_DETECTION_MODEL.detect_face(img)
    return res


def detect_emotion(img):
    res = EMOTIONS_DETECTION_MODEL.detect_face(img)
    return res


def compare_faces(user_id, actor_id):
    res = COMPARISON_DETECTION_MODEL.compare_faces(user_id, actor_id)
    return res


def get_only_needed_keypoints(keypoints):
    return [keypoints['mouthLeft'], keypoints['mouthRight'], keypoints['eyebrowLeftOuter'],
            keypoints['eyebrowLeftInner'], keypoints['eyebrowRightInner'],
            keypoints['eyebrowRightOuter']]


@app.route('/emotion', methods=['POST'])
def emotions():
    if 'file' not in request.files:
        return jsonify(error="Please try again. The Image doesn't exist")

    file = request.files.get('file')

    if not file:
        return jsonify(error="Please try again. The Image doesn't exist")

    try:
        response_json = detect_emotion(file)
        face_emotions = response_json["faceAttributes"]["emotion"]
        face_rectangle = response_json["faceRectangle"]
        return jsonify(emotions=face_emotions, rectangle=face_rectangle)
    except Exception as e:
        return jsonify(error=e)


@app.route('/keypoints', methods=['POST'])
def keypoints():
    if 'file_user' not in request.files or 'file_actor' not in request.files:
        return jsonify(error="Please try again. The Image doesn't exist")

    file_user = request.files.get('file_user')
    file_actor = request.files.get('file_actor')

    try:
        response_user = detect_keypoints(file_user)
        response_actor = detect_keypoints(file_actor)
        keypoints_user = response_user['faceLandmarks']
        keypoints_actor = response_actor['faceLandmarks']

        only_needed_keypoints_user = get_only_needed_keypoints(keypoints_user)
        only_needed_keypoints_actor = get_only_needed_keypoints(keypoints_actor)

        face_rectangle_user = response_user['faceRectangle']
        face_rectangle_actor = response_actor['faceRectangle']

        return jsonify(keypoints_user=only_needed_keypoints_user, keypoints_actor=only_needed_keypoints_actor,
                       rectangle_user=face_rectangle_user,
                       rectangle_actor=face_rectangle_actor)

    except Exception as e:
        return jsonify(error=str(e))


@app.route('/compare', methods=['POST'])
def compare():
    if 'file_user' not in request.files or 'file_actor' not in request.files:
        return jsonify(error="Please try again. The Image doesn't exist")

    file_user = request.files.get('file_user')
    file_actor = request.files.get('file_actor')

    try:

        response_user = detect_id(file_user)
        response_actor = detect_id(file_actor)

        id_user = response_user["faceId"]
        id_actor = response_actor["faceId"]

        face_rectangle_user = response_user['faceRectangle']
        face_rectangle_actor = response_actor['faceRectangle']

        comparison_results = compare_faces(id_user, id_actor)

        return jsonify(comparison_results=comparison_results, rectangle_user=face_rectangle_user,
                       rectangle_actor=face_rectangle_actor)

    except Exception as e:
        return jsonify(error=str(e))


@app.route('/', methods=['GET'])
def index():
    return 'Bad way'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
