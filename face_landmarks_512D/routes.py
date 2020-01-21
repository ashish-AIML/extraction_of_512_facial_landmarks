# import faiss
import cv2
import uuid
import face_model
import pika
import json
import base64
import numpy as np
from py_common import tMQ
import uuid

# load model 
det=0
gpu=0
image_size='112,112'
model='/mxnet/model-r100-ii/model,0'
threshold=0.9
loder=[det,gpu,image_size,model,threshold]
clf = None
ch = None

X = []
y = []

# load face recognition model 
model = face_model.FaceModel(loder) 

def _detector(img):
    global model
    faces = model.get_input(img)
    return faces

def _extractor(face):
    face_coordinates = model.get_feature(face)
    return face_coordinates

def extractFaceCoordinates(ch, props, reqBody):

    exchange = reqBody["exchange"]
    type = str(reqBody['type'])
    
    if type == "frame":
        # content = base64.b64decode(content)
        trackerID = reqBody['id']
        content = str(reqBody['content'])
        nparr = np.fromstring(content.decode('base64'), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

    elif type == "file_path":
        # content = base64.b64decode(content)
        trackerID = "123"
        file_path = str(reqBody['file_path'])
        print("file_path -->",file_path)
        img = cv2.imread(file_path)
    
    # filename = "faces/"+str(uuid.uuid4())+".jpg"
    # with open(filename, 'wb') as f:
    #     f.write(content)
    # detect faces
    # img = cv2.imread(filename)

    faces = _detector(img)
    if faces == None or len(faces) == 0:
        print("no faces ...")
        if props.reply_to:
            tMQ.ReplyBack(props,{"http_code":400})
        else:
           tMQ.Execute(exchange,"face.unknown",{"http_code":400,"data":{"id":trackerID}})
        # tMQ.RaiseEvent("face.extraction.failed",{"http_code":200,"data":{"id":trackerID}})
    else:
        # extract coordinates of first face
        face = faces[0]
        face_coordinates = _extractor(face)

        if props.reply_to:
            tMQ.ReplyBack(props,{"http_code":200,"data":{"coordinates":face_coordinates.tolist()}})
        else:
           tMQ.Execute(exchange,"face.recognise",{"http_code":200,"data":{"id":trackerID,"coordinates":face_coordinates.tolist()}})
        # tMQ.RaiseEvent("face.extraction.success",{"http_code":200,"data":{"id":trackerID,"coordinates":face_coordinates.tolist()}})

    # imgBase64 = reqBody["content"]
    # trackerID = reqBody["tracker_id"]
    # imgdata = base64.b64decode(reqBody["content"])

    # filename = "faces/"+str(uuid.uuid4())+".jpg"
    # with open(filename, 'wb') as f:
    #     f.write(imgdata)

    # # extract coordinates
    # # _replyBack({"data":{"coordinates":face_coordinates.tolist(),"file_name":filename}},200,properties)
    # face = cv2.imread(filename)
    # face_coordinates = model.get_feature(face)
    # tMQ.RaiseEvent("face.extracted",{"http_code":200,"success":True,"data":{"tracker_id":trackerID,"coordinates":face_coordinates.tolist()}})