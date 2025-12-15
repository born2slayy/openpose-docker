

from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import base64
from openpose import pyopenpose as op

app = FastAPI()

# OpenPose params
params = {
    "model_folder": "/usr/local/openpose/models/",
    "number_people_max": 1,
    "face": True,
    "hand": False,

}


opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def safe_np(kp):

    if kp is None:
        return []
    try:
        return kp.tolist()
    except:
        return []

@app.post("/pose/")
async def get_pose(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    datum = op.Datum()
    datum.cvInputData = image

    vec = op.VectorDatum()
    vec.append(datum)

    # OpenPose 
    opWrapper.emplaceAndPop(vec)

    # Body keypoints
    body = safe_np(datum.poseKeypoints)
    persons = len(body) if isinstance(body, list) else 0

    # Face keypoints
    face = safe_np(datum.faceKeypoints)

    # Hand keypoints
    left = []
    right = []
    if datum.handKeypoints is not None:
        left = safe_np(datum.handKeypoints[0]) if len(datum.handKeypoints) > 0 else []
        right = safe_np(datum.handKeypoints[1]) if len(datum.handKeypoints) > 1 else []

    # Rendered image (skeleton)
    rendered = ""
    if datum.cvOutputData is not None:
        _, buffer = cv2.imencode(".jpg", datum.cvOutputData)
        rendered = base64.b64encode(buffer).decode("utf-8")

    return {
        "persons": persons,
        "body_keypoints": body,
        "face_keypoints": face,
        "left_hand": left,
        "right_hand": right,
        "image_shape": image.shape,
        "rendered_image": rendered,
    }


