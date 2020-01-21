# import faiss
import cv2

import face_model

import numpy as np
import mxnet

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

# def extractFaceCoordinates(ch, props, reqBody):

        
    file_path = str(reqBody['file_path'])
        
    img = cv2.imread(file_path)
    
    # filename = "faces/"+str(uuid.uuid4())+".jpg"
    # with open(filename, 'wb') as f:
    #     f.write(content)
    # detect faces
    # img = cv2.imread(filename)

    faces = _detector(img)
    if faces == None or len(faces) == 0:
        print("no faces ...")
    else:
        # extract coordinates of first face
        face = faces[0]
        face_coordinates = _extractor(face)