import faiss
import os
import cv2
import pika
import numpy as np
import json # convert input or response to/from JSON
import uuid # to store image recieved from RabbitMQ
import base64 # to decode image coming from RabbitMQ
from os import path

X = []
y = []



faissMappedPeople = {}
# load faiss mapping person_id
if(path.exists("faissMappedPeople.json")):
    with open("faissMappedPeople.json", "r") as write_file:
        faissMappedPeople = json.load(write_file)

# helper functions -- 
# for KNN we had to retrain all faces
# # for Faiss we can add new faces to existing model
def trainFaissModel(X):
    model_save_path = "faiss_model"
    if os.path.isfile(model_save_path + '/512_model'):
        index = faiss.read_index(model_save_path + '/512_model')
    else:
        print('faiss index is not available, creating index')
        index = faiss.IndexFlatL2(512)  # build the index size of vector
        index = faiss.IndexIDMap(index)
    
    index.reset()
    faiss_index = 1

    for i, x in enumerate(X):
        xb = np.zeros((1, 512)).astype('float32')
        embedding = np.array(list(x)).astype(np.float)
        xb[0] = embedding
        ids = np.arange(1) + faiss_index
        index.add_with_ids(xb, ids)
        faiss_index += 1 # index is auto calculated
    
    faiss.write_index(index, model_save_path + '/512_model')
    print("new face model saved with "+str(len(X))+" faces")
    # print(y)

def _detector(img):
    faces = model.get_input(img)
    return faces

def _extractor(face):
    face_coordinates = model.get_feature(face)
    return face_coordinates

def _recognizer(face_coordinates,k=4):
    model_save_path = "faiss_model"
    index = faiss.read_index(model_save_path + '/512_model')
    xb = np.zeros((1, 512)).astype('float32') 
    xb[0] = face_coordinates
    D, I = index.search(xb, k)  # actual search
    k1, k2, k3, k4 = I[0][0], I[0][1], I[0][2], I[0][3]
    result = [k1, k2, k3, k4]

    # get the mapping again -- optimize later
    if(path.exists("faissMappedPeople.json")):
        with open("faissMappedPeople.json", "r") as write_file:
            faissMappedPeople = json.load(write_file)

    print("start",result,D,faissMappedPeople,"end")

    if D[0][0] <= 0.9:
        strk1 = str(k1)
        if strk1 in faissMappedPeople:
            if D[0][0] <= 0.2:
                accuracy = "0.95"
            elif D[0][0] <= 0.4:
                accuracy = "0.90"
            elif D[0][0] <= 0.6:
                accuracy = "0.80"
            else:
                accuracy = ( 0.9 - D[0][0] ) # 1 - distance | 1 - 0.4 distance
            return True, faissMappedPeople[strk1], accuracy
        else:
            return False, None, 0
    else:
        return False, None, 0

    # global clf
    # rec = None
    # pred = None
    # # load KNN model
    # knn_model_path="tericsoft.clf"
    # if clf is None:
    #     with open(knn_model_path, 'rb') as f:
    #         clf = pickle.load(f)
    # # format face cordinates to feed into KNN
    # expanded_face_coordinates = np.expand_dims(face_coordinates, axis=0)
    # closest_distances = clf.kneighbors(expanded_face_coordinates, n_neighbors=3)
    # X_face_locations=[(206, 954, 527, 9000)]
    # are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(closest_distances[0]))]
    # print([(closest_distances[0][0][0], pred, loc) if rec else (closest_distances[0][0][0], "unknown", loc) for pred, loc, rec in zip(clf.predict(expanded_face_coordinates), X_face_locations, are_matches)])
    # if rec:
    #     return True, pred, closest_distances[0][0][0]
    # else:
    #     return False, None, 0
    return False, None, 0

def _replyBack(response,http_code, props):
    # data = {"data":data,"http_code":http_code}
    response["http_code"] = http_code
    jsonStr = json.dumps(response)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id = props.correlation_id),
                     body=str(jsonStr))
    print("replying ...",props.reply_to,props.correlation_id)

def _raiseEvent(pattern,data):
    jsonStr = json.dumps(data)
    ch.basic_publish(exchange='events',
                     routing_key=pattern,
                     body=str(jsonStr))
    print("raising event ...",pattern)

# only extract coordinates
def extractImage(ch, properties, reqBody):
    # get image
    reqBody = json.loads(reqBody)
    imgdata = base64.b64decode(reqBody["content"])
    filename = "faces/"+str(uuid.uuid4())+".jpg"
    with open(filename, 'wb') as f:
        f.write(imgdata)

    # detect faces
    img = cv2.imread(filename)
    faces = _detector(img)
    if faces == None or len(faces) == 0:
        os.remove(filename) # delete image recieved 
        _replyBack({"errors":{"face":"no face found"}}, 400,properties)
        return
    
    # extract coordinates of first face
    face = faces[0]
    face_coordinates = _extractor(face)
    os.remove(filename) # delete image recieved 
    _replyBack({"data":{"coordinates":face_coordinates.tolist(),"file_name":filename}},200,properties)

# retrain X, y KNN
def train(ch, properties, reqBody):
    X = []
    # y = []
    trained_images = []
    faissMappedPeople = {}
    reqBody = json.loads(reqBody)

    if "personImages" in reqBody and len(reqBody["personImages"]) > 0:
        faiss_index = 1
        for image in reqBody["personImages"]:
            coordinates = image["coordinates"]
            trained_images.append(image["_id"])
            X.append(coordinates)
            faissMappedPeople[faiss_index] = image["person_id"]
            faiss_index += 1
        print("faissMappedPeople before calling train -->",faissMappedPeople)

        with open("faissMappedPeople.json", "w") as write_file:
            json.dump(faissMappedPeople, write_file)
        
        trainFaissModel(X)

        _raiseEvent("training.completed",trained_images)

    else:
        print("recieved 0 faces -- not training")

# extract and give name
def recognize(ch, properties, reqBody):
    # get image
    print("calling recognize ..")
    reqBody = json.loads(reqBody)
    imgdata = base64.b64decode(reqBody["content"])
    filename = "faces/"+str(uuid.uuid4())+".jpg"
    with open(filename, 'wb') as f:
        f.write(imgdata)

    # detect faces
    img = cv2.imread(filename)
    faces = _detector(img)
    if faces == None or len(faces) == 0:
        os.remove(filename) # delete image recieved 
        _replyBack({"error":"no face found ding dong"},500,properties)
        return

    # extract coordinates of first face
    matches = []
    for face in faces:
        face_coordinates = _extractor(face)
        # recognise
        face_matched, person_id,accuracy = _recognizer(face_coordinates,4)
        if face_matched:
            matches.append({"person_id":person_id,"accuracy":str(accuracy)}) 
        
    if len(matches) > 0:
        print("replying matches = ",matches)
        _replyBack({"data":matches},200,properties)
        return
    else:
        print("matches count = 0 so sending 400")
        _replyBack({"data":None},400,properties)
        return

