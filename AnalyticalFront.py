import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import time
import numpy as np
from Tracker.centroidtracker import CentroidTracker
import os
import random
import requests
import zmq
from multiprocessing import Process

faceProto = "face_deploy.prototxt"
faceModel = "face_net.caffemodel"
faceNet = cv2.dnn.readNetFromCaffe(faceProto, faceModel)
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
ageNet = cv2.dnn.readNetFromCaffe(ageProto, ageModel)

def process2():
    while True:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        message = socket.recv_string()
        return message

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

genderList = ['Male', 'Female']
ageList = ['Kids', 'Kids', 'Young', 'Young', 'Adult', 'Adult', 'Senior', 'Senior']

date = time.strftime('%Y-%m-%d')

robotID = 1
maxSessID = 50

cam = 0
camStatus = 'Front'

seconds = minutes = hours = 0

video = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
# video.set(3, 960)
# video.set(4, 540)
video.set(3, 640)
video.set(4, 480)

if cam == 2:
    cam = 1
else:
    cam = 2

data_path = f"C:\MY FILE\CODE\Analytical\Dataset\Person_Analyze"
pic_path = f"C:\MY FILE\CODE\Analytical\Dataset\Person_Analyze"

isExist_data  = os.path.exists(data_path)
isExist_pic  = os.path.exists(pic_path)

if not isExist_data:
    data_path = f"Person_Analyze/{date}/{camStatus}/Data"
    isDir_data = os.path.isdir(data_path)
    if not isDir_data:
        os.makedirs(data_path)
    listDir = os.listdir(data_path)
    if len(listDir) != 0:
        listDir_order = int(listDir[-1].split('_')[2])
    else:
        listDir_order = 0
else:
    data_path = f"{data_path}/{date}/{camStatus}/Data"
    isDir_data = os.path.isdir(data_path)
    if not isDir_data:
        os.makedirs(data_path)
    listDir = os.listdir(data_path)
    if len(listDir) != 0:
        listDir_order = int(listDir[-1].split('_')[2])
    else:
        listDir_order = 0

if not isExist_pic:
    pic_path = f"Person_Analyze/{date}/{camStatus}/Picture"
    isDir_pic = os.path.isdir(pic_path)
    if not isDir_pic:
        os.makedirs(pic_path)
else:
    pic_path = f"{pic_path}/{date}/{camStatus}/Picture"
    isDir_pic = os.path.isdir(pic_path)
    if not isDir_pic:
        os.makedirs(pic_path)

detector = FaceMeshDetector()
ct = CentroidTracker()

fontStyle = cv2.FONT_HERSHEY_DUPLEX
fontColor = (0, 0, 0)
fontSize = 0.45

confidence = 0.5

idxSet = set()
idxSetOn = set()

csvList = list()
csvList_status = True

partData = 0
countList = 0
sessID = -1

noFace_status = True
startProgram = True
prevStart_time = 0

genderID = [
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    ]
ageID = [
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    ]
durationDataID = [
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], 
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], 
    ]
interestDataID = [
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], 
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], 
    ]
timeID = [
    ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], 
    ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], 
    ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], 
    ]
pic_name = [
    ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], 
    ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], 
    ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], 
    ]

attTrue = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    ]
    
attFalse = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    ]

start_durationAtt = [
    [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], 
    [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], 
    [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], 
    ]

genderPredID = [
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    ]

agePredID = [
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    ]

start_durationPred = [
    [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], 
    [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], 
    [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], 
    ]

writeStatus = [
    [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
    [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
    [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
    ]

session = ''.join(random.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') for i in range(16))

payload = {
  'sess_id': 0, # integer
  'person_id': 0, # integer
  'gender': '', # string
  'age': '', # string
  'att_dur': 0.0, # float
  'interest': 0, # integer
  'date': "YYYY-MM-DD", # string
  'time': "hh:mm:ss", # string
  'cam': 0, # integer
  'filename': "xxx.jpg", # string
  'session': "xxxxxxxxxxxxxxxx", # string
  'robot_id': 1 # integer
}

def process1():
    while True:
        startTime = time.time()
        count = round(startTime, 2)
        if startProgram:
            startDuration = count
            startProgram = False
        durationMaster = round((count - startDuration), 2)
        ret, frame = video.read()
        if not ret:
            print("Camera disconnected")
            break
        
        partData = countList // maxSessID
        files = f"{data_path}/data_{date}_{str(listDir_order+1).zfill(3)}_{partData}.csv"
        txt_file = open(files, "a+")
        if countList % maxSessID == 0:
            if csvList_status == True:
                csvList.append(files)
                txt_file.write('Session ID,Person ID,Gender,Age,Attention Duration,Interest,Date,Time,Files Name\n')
                csvList_status = False
        else:
            csvList_status = True

        H, W = frame.shape[:2]
        # h_resize, w_resize = round(H*0.5), round(W*0.5)
        # frame = cv2.resize(frame, (w_resize, h_resize))
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        faceNet.setInput(blob)
        detections = faceNet.forward()
        detect_object = detections[0,0,:,1]
        length_of_object = len(detect_object)
        rects = [] # untuk object tracker
        for i in range(0, detections.shape[2]):
            if detections[0, 0, i, 2] > confidence:
                new_image = frame.copy()
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(box.astype("int"))
                
        objects = ct.update(rects) # baca object tracker
        if len(objects) > 0:
            noFace_status = True
            idxSetOn = set(list(objects.keys()))
            for (objectID, centroid) in objects.items():
                # rumus unique id
                index = objectID % 30
                indexPerson = objectID+1
                indexReset = index + 15
                if indexReset > 29:
                    indexReset = indexReset - 30
                
                # RESET DATA
                durationDataID[indexReset] = [0]
                interestDataID[indexReset] = [0]
                timeID[indexReset] = ["", True]
                pic_name[indexReset] = ["", True]
                attTrue[indexReset] = 1
                attFalse[indexReset] = 1
                start_durationAtt[indexReset] = [0, True]
                genderPredID[indexReset] = []
                agePredID[indexReset] = []
                start_durationPred[indexReset] = [0, True]
                writeStatus[indexReset] = [True]

                # face detection
                startX, startY, endX, endY = centroid[2], centroid[3], centroid[4], centroid[5]
                h_rect = endX - startX
                ratioFont = h_rect/h_rect*0.5
                text = "Person {}".format(str(indexPerson))
                cv2.putText(frame, text, (centroid[2], centroid[3] - 10), fontStyle, ratioFont, (0, 255, 0), 1)
                boxSize = (endX - startX)*(endY - startY)

                # DEFINE ROI (area crop untuk penyimpanan gambar yang terdeteksi dan pendeteksian umur & jenis kelamin)
                biasX = round((endX - startX)*0.35)
                biasY = round((endY - startY)*0.35)
                xLeft, yLeft, xRight, yRight = startX-biasX, startY-biasY, endX+biasX, endY+biasY
                if xLeft < 0:
                    xLeft = 0
                if yLeft < 0:
                    yLeft = 0
                if xRight > W:
                    xRight = W
                if yRight > H:
                    yRight = H
                ROI = new_image[yLeft:yRight, xLeft:xRight]
                resizedROI_large = cv2.resize(ROI, (ROI.shape[1]*6, ROI.shape[0]*6))
                resizedROI_medium = cv2.resize(ROI, (ROI.shape[1]*4, ROI.shape[0]*4))
                resizedROI_small = cv2.resize(ROI, (ROI.shape[1]*2, ROI.shape[0]*2))
                image, faces = detector.findFaceMesh(ROI, draw=True) # mendeteksi face mesh dari ROI

                if faces: # jika face mesh terdeteksi
                    # DEFINE DOTS
                    face = faces[0]
                    leftSideA, leftSideB, rightSideA, rightSideB, center = face[356], face[359], face[127], face[33], face[6]
                    upperSideA, upperSideB, bottomSideA, bottomSideB = face[10], face[151], face[200], face[152]
                    areaUpper = round(detector.findDistance(upperSideA, upperSideB)[0])
                    areaBottom = round(detector.findDistance(bottomSideA, bottomSideB)[0])
                    areaLeft = round(detector.findDistance(leftSideB, center)[0])
                    areaRight = round(detector.findDistance(rightSideB, center)[0])

                    # CALCULATE THE DEVIATIONS (hitung penyimpangan)
                    deviationHorizontal = abs(areaLeft - areaRight)
                    deviationVertical = abs(areaUpper - areaBottom)
                    # print(f'Box Size: {boxSize}, Deviation Horizontal: {deviationHorizontal}, Deviation Vertical: {deviationVertical}')

                    # Timer saat terdapat perhatian (attention)
                    if start_durationPred[index][1]:
                        start_durationPred[index][0] = durationMaster
                        start_durationPred[index][1] = False
                    durationPred = round((durationMaster - start_durationPred[index][0]), 2)

                    if durationPred < 5:
                        # pendeteksian umur dan jenis kelamin
                        blob = cv2.dnn.blobFromImage(ROI, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                        genderNet.setInput(blob)
                        ageNet.setInput(blob)
                        genderPreds = genderNet.forward()
                        gender = genderList[genderPreds[0].argmax()]
                        agePreds = ageNet.forward()
                        age = ageList[agePreds[0].argmax()]
                        genderPredID[index].append(gender)
                        agePredID[index].append(age)
                        genderID[index] = max(set(genderPredID[index]), key=genderPredID[index].count)
                        ageID[index] = max(set(agePredID[index]), key=agePredID[index].count)
                        startX_stats, endX_stats, startY_stats, endY_stats = startX, startX + 120, endY - 1, endY + 50
                        if startX_stats < 0:
                            startX_stats = 0
                        if startY_stats < 0:
                            startY_stats = 0
                        if endX_stats > W:
                            endX_stats = W
                        if endY_stats > H:
                            endY_stats = H
                        
                        # display di frame
                        sub_frame = frame[startY_stats:endY_stats, startX_stats:endX_stats]
                        white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                        res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
                        frame[startY_stats:endY_stats, startX_stats:endX_stats] = res
                        
                    else:
                        genderID[index] = max(set(genderPredID[index]), key=genderPredID[index].count)
                        ageID[index] = max(set(agePredID[index]), key=agePredID[index].count)
                    
                    finalGender = genderID[index]
                    finalAge = ageID[index]
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), int(round(H/320)))
                    cv2.putText(frame, f'Gender: {finalGender}', (startX+2, endY+15), fontStyle, fontSize, fontColor, 1)
                    cv2.putText(frame, f'Age: {finalAge}', (startX+2, endY+30), fontStyle, fontSize, fontColor, 1)
                    if boxSize < 40000:
                        if deviationHorizontal < 10 and deviationVertical < 15:
                            if timeID[index][1]:
                                timeID[index][0] = time.strftime('%T')
                                timeID[index][1] = False
                            if start_durationAtt[index][1]:
                                start_durationAtt[index][0] = durationMaster
                                start_durationAtt[index][1] = False
                            durationAtt = durationDataID[index][attTrue[index]-1] = round((durationMaster - start_durationAtt[index][0]), 2)
                            str_durationAtt = str(durationAtt)
                            startX_stats, endX_stats, startY_stats, endY_stats = startX, startX + 120, endY - 1, endY + 50
                            if startX_stats < 0:
                                startX_stats = 0
                            if startY_stats < 0:
                                startY_stats = 0
                            if endX_stats > W:
                                endX_stats = W
                            if endY_stats > H:
                                endY_stats = H
                            sub_frame = frame[startY_stats:endY_stats, startX_stats:endX_stats]
                            white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                            res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
                            frame[startY_stats:endY_stats, startX_stats:endX_stats] = res 
                            cv2.putText(frame, f'Gender: {finalGender}', (startX+2, endY+15), fontStyle, fontSize, fontColor, 1)
                            cv2.putText(frame, f'Age: {finalAge}', (startX+2, endY+30), fontStyle, fontSize, fontColor, 1)
                            cv2.putText(frame, str_durationAtt, (startX+2, endY+45), fontStyle, fontSize, fontColor, 1)            
                            if durationAtt >= 0.1:
                                if pic_name[index][1]:
                                    pic_name[index][0] = f"{date}_{str(listDir_order+1).zfill(3)}_id{indexPerson}_att{attTrue[index]}.jpg"
                                    cv2.imwrite(f"{pic_path}/{pic_name[index][0]}", resizedROI_large)
                                    print(f"Face detected! Picture saved as {pic_name[index][0]}")
                                if attFalse[index] == attTrue[index]:
                                    attFalse[index] += 1
                                    idxSet.add(indexPerson - 1)
                                pic_name[index][1] = False

                            else:
                                pic_name[index] = ["", True]
                            
                        else:
                            timeID[index][1] = True
                            if attTrue[index] != attFalse[index]:
                                attTrue[index] += 1
                            startX_stats, endX_stats, startY_stats, endY_stats = startX, startX + 120, endY - 1, endY + 50
                            if startX_stats < 0:
                                startX_stats = 0
                            if startY_stats < 0:
                                startY_stats = 0
                            if endX_stats > W:
                                endX_stats = W
                            if endY_stats > H:
                                endY_stats = H
                            start_durationAtt[index][1] = True
                            sub_frame = frame[startY_stats:endY_stats, startX_stats:endX_stats]
                            white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                            res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
                            frame[startY_stats:endY_stats, startX_stats:endX_stats] = res
                            cv2.putText(frame, f'Gender: {finalGender}', (startX+2, endY+15), fontStyle, fontSize, fontColor, 1)
                            cv2.putText(frame, f'Age: {finalAge}', (startX+2, endY+30), fontStyle, fontSize, fontColor, 1)

                    if 40000 <= boxSize <= 55000:
                        if deviationHorizontal < 20 and deviationVertical < 15:
                            if timeID[index][1]:
                                timeID[index][0] = time.strftime('%T')
                                timeID[index][1] = False
                            if start_durationAtt[index][1]:
                                start_durationAtt[index][0] = durationMaster
                                start_durationAtt[index][1] = False
                            durationAtt = durationDataID[index][attTrue[index]-1] = round((durationMaster - start_durationAtt[index][0]), 2)
                            str_durationAtt = str(durationAtt)
                            startX_stats, endX_stats, startY_stats, endY_stats = startX, startX + 120, endY - 1, endY + 50
                            if startX_stats < 0:
                                startX_stats = 0
                            if startY_stats < 0:
                                startY_stats = 0
                            if endX_stats > W:
                                endX_stats = W
                            if endY_stats > H:
                                endY_stats = H
                            sub_frame = frame[startY_stats:endY_stats, startX_stats:endX_stats]
                            white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                            res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
                            frame[startY_stats:endY_stats, startX_stats:endX_stats] = res 
                            cv2.putText(frame, f'Gender: {finalGender}', (startX+2, endY+15), fontStyle, fontSize, fontColor, 1)
                            cv2.putText(frame, f'Age: {finalAge}', (startX+2, endY+30), fontStyle, fontSize, fontColor, 1)
                            cv2.putText(frame, str_durationAtt, (startX+2, endY+45), fontStyle, fontSize, fontColor, 1)
                            if durationAtt >= 0.1:
                                if pic_name[index][1]:
                                    pic_name[index][0] = f"{date}_{str(listDir_order+1).zfill(3)}_id{indexPerson}_att{attTrue[index]}.jpg"
                                    cv2.imwrite(f"{pic_path}/{pic_name[index][0]}", resizedROI_medium)
                                    print(f"Face detected! Picture saved as {pic_name[index][0]}")
                                if attFalse[index] == attTrue[index]:
                                    attFalse[index] += 1
                                    idxSet.add(indexPerson - 1)
                                pic_name[index][1] = False


                            else:
                                pic_name[index] = ["", True]
                            
                        else:
                            timeID[index][1] = True
                            if attTrue[index] != attFalse[index]:
                                attTrue[index] += 1
                            startX_stats, endX_stats, startY_stats, endY_stats = startX, startX + 120, endY - 1, endY + 50
                            if startX_stats < 0:
                                startX_stats = 0
                            if startY_stats < 0:
                                startY_stats = 0
                            if endX_stats > W:
                                endX_stats = W
                            if endY_stats > H:
                                endY_stats = H
                            start_durationAtt[index][1] = True
                            sub_frame = frame[startY_stats:endY_stats, startX_stats:endX_stats]
                            white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                            res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
                            frame[startY_stats:endY_stats, startX_stats:endX_stats] = res
                            cv2.putText(frame, f'Gender: {finalGender}', (startX+2, endY+15), fontStyle, fontSize, fontColor, 1)
                            cv2.putText(frame, f'Age: {finalAge}', (startX+2, endY+30), fontStyle, fontSize, fontColor, 1)
                        
                    if 55000 < boxSize <= 70000:
                        if deviationHorizontal < 30 and deviationVertical < 15:
                            if timeID[index][1]:
                                timeID[index][0] = time.strftime('%T')
                                timeID[index][1] = False
                            if start_durationAtt[index][1]:
                                start_durationAtt[index][0] = durationMaster
                                start_durationAtt[index][1] = False
                            durationAtt = durationDataID[index][attTrue[index]-1] = round((durationMaster - start_durationAtt[index][0]), 2)
                            str_durationAtt = str(durationAtt)
                            startX_stats, endX_stats, startY_stats, endY_stats = startX, startX + 120, endY - 1, endY + 50
                            if startX_stats < 0:
                                startX_stats = 0
                            if startY_stats < 0:
                                startY_stats = 0
                            if endX_stats > W:
                                endX_stats = W
                            if endY_stats > H:
                                endY_stats = H
                            sub_frame = frame[startY_stats:endY_stats, startX_stats:endX_stats]
                            white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                            res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
                            frame[startY_stats:endY_stats, startX_stats:endX_stats] = res 
                            cv2.putText(frame, f'Gender: {finalGender}', (startX+2, endY+15), fontStyle, fontSize, fontColor, 1)
                            cv2.putText(frame, f'Age: {finalAge}', (startX+2, endY+30), fontStyle, fontSize, fontColor, 1)
                            cv2.putText(frame, str_durationAtt, (startX+2, endY+45), fontStyle, fontSize, fontColor, 1)
                            if durationAtt >= 0.1:
                                if pic_name[index][1]:
                                    pic_name[index][0] = f"{date}_{str(listDir_order+1).zfill(3)}_id{indexPerson}_att{attTrue[index]}.jpg"
                                    cv2.imwrite(f"{pic_path}/{pic_name[index][0]}", resizedROI_small)
                                    print(f"Face detected! Picture saved as {pic_name[index][0]}")
                                if attFalse[index] == attTrue[index]:
                                    attFalse[index] += 1
                                    idxSet.add(indexPerson - 1)
                                pic_name[index][1] = False

                            else:
                                pic_name[index] = ["", True]
                                
                        else:
                            timeID[index][1] = True
                            if attTrue[index] != attFalse[index]:
                                attTrue[index] += 1
                            startX_stats, endX_stats, startY_stats, endY_stats = startX, startX + 120, endY - 1, endY + 50
                            if startX_stats < 0:
                                startX_stats = 0
                            if startY_stats < 0:
                                startY_stats = 0
                            if endX_stats > W:
                                endX_stats = W
                            if endY_stats > H:
                                endY_stats = H
                            start_durationAtt[index][1] = True
                            sub_frame = frame[startY_stats:endY_stats, startX_stats:endX_stats]
                            white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                            res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
                            frame[startY_stats:endY_stats, startX_stats:endX_stats] = res
                            cv2.putText(frame, f'Gender: {finalGender}', (startX+2, endY+15), fontStyle, fontSize, fontColor, 1)
                            cv2.putText(frame, f'Age: {finalAge}', (startX+2, endY+30), fontStyle, fontSize, fontColor, 1)
                        
                    if boxSize > 70000:
                        if deviationHorizontal < 40 and deviationVertical < 15:
                            if timeID[index][1]:
                                timeID[index][0] = time.strftime('%T')
                                timeID[index][1] = False
                            if start_durationAtt[index][1]:
                                start_durationAtt[index][0] = durationMaster
                                start_durationAtt[index][1] = False
                            durationAtt = durationDataID[index][attTrue[index]-1] = round((durationMaster - start_durationAtt[index][0]), 2)
                            str_durationAtt = str(durationAtt)
                            startX_stats, endX_stats, startY_stats, endY_stats = startX, startX + 120, endY - 1, endY + 50
                            if startX_stats < 0:
                                startX_stats = 0
                            if startY_stats < 0:
                                startY_stats = 0
                            if endX_stats > W:
                                endX_stats = W
                            if endY_stats > H:
                                endY_stats = H
                            sub_frame = frame[startY_stats:endY_stats, startX_stats:endX_stats]
                            white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                            res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
                            frame[startY_stats:endY_stats, startX_stats:endX_stats] = res 
                            cv2.putText(frame, f'Gender: {finalGender}', (startX+2, endY+15), fontStyle, fontSize, fontColor, 1)
                            cv2.putText(frame, f'Age: {finalAge}', (startX+2, endY+30), fontStyle, fontSize, fontColor, 1)
                            cv2.putText(frame, str_durationAtt, (startX+2, endY+45), fontStyle, fontSize, fontColor, 1)
                            if durationAtt >= 0.1:
                                if pic_name[index][1]:
                                    pic_name[index][0] = f"{date}_{str(listDir_order+1).zfill(3)}_id{indexPerson}_att{attTrue[index]}.jpg"
                                    cv2.imwrite(f"{pic_path}/{pic_name[index][0]}", ROI)
                                    print(f"Face detected! Picture saved as {pic_name[index][0]}")
                                if attFalse[index] == attTrue[index]:
                                    attFalse[index] += 1
                                    idxSet.add(indexPerson - 1)
                                pic_name[index][1] = False

                            else:
                                pic_name[index] = ["", True]
                        
                        else:
                            timeID[index][1] = True
                            if attTrue[index] != attFalse[index]:
                                attTrue[index] += 1
                            startX_stats, endX_stats, startY_stats, endY_stats = startX, startX + 120, endY - 1, endY + 50
                            if startX_stats < 0:
                                startX_stats = 0
                            if startY_stats < 0:
                                startY_stats = 0
                            if endX_stats > W:
                                endX_stats = W
                            if endY_stats > H:
                                endY_stats = H
                            start_durationAtt[index][1] = True
                            sub_frame = frame[startY_stats:endY_stats, startX_stats:endX_stats]
                            white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                            res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
                            frame[startY_stats:endY_stats, startX_stats:endX_stats] = res
                            cv2.putText(frame, f'Gender: {finalGender}', (startX+2, endY+15), fontStyle, fontSize, fontColor, 1)
                            cv2.putText(frame, f'Age: {finalAge}', (startX+2, endY+30), fontStyle, fontSize, fontColor, 1)
                                
                else:
                    timeID[index][1] = True
                    start_durationAtt[index][1] = True
                    genderPredID[index] = []
                    agePredID[index] = []
                    start_durationPred[index][1] = True

                if len(durationDataID[index]) == attTrue[index]-1:
                    durationDataID[index].append(0)
                if len(interestDataID[index]) == attTrue[index]-1:
                    interestDataID[index].append(0)
                if len(writeStatus[index]) == attTrue[index]-1:
                    writeStatus[index].append(True)

                if durationDataID[index][attTrue[index]-1] >= 0.1 and durationDataID[index][attTrue[index]-1] < 2.5:
                    interestDataID[index][attTrue[index]-1] = 0
                if durationDataID[index][attTrue[index]-1] >= 2.5 and durationDataID[index][attTrue[index]-1] < 5:
                    interestDataID[index][attTrue[index]-1] = 1
                if durationDataID[index][attTrue[index]-1] >= 5:
                    interestDataID[index][attTrue[index]-1] = 2

                idxSetOff = idxSet.difference(idxSetOn)

                if len(writeStatus[index]) > 1:
                    if writeStatus[index][attTrue[index]-2] == True:
                        sessID += 1
                        countList += 1
                        print(f'{sessID}|{indexPerson}|{genderID[index]}|{ageID[index]}|{durationDataID[index][attTrue[index]-2]}|{interestDataID[index][attTrue[index]-2]}|{date}|{timeID[index][0]}|{pic_name[index][0]}')
                        txt_file.write(f'{sessID},{indexPerson},{genderID[index]},{ageID[index]},{durationDataID[index][attTrue[index]-2]},{interestDataID[index][attTrue[index]-2]},{date},{timeID[index][0]},{pic_name[index][0]}\n')
                        payload["sess_id"] = sessID
                        payload["person_id"] = indexPerson
                        payload["gender"] = genderID[index]
                        payload["age"] = ageID[index]
                        payload["att_dur"] = durationDataID[index][attTrue[index]-2]
                        payload["interest"] = interestDataID[index][attTrue[index]-2]
                        payload["date"] = date
                        payload["time"] = timeID[index][0]
                        payload["cam"] = cam
                        payload["filename"] = pic_name[index][0]
                        payload["session"] = session
                        payload["robot_id"] = robotID
                        writeStatus[index][attTrue[index]-2] = False
                        try:
                            res = requests.post('https://ropi.web.id/api/pos.php', data=payload)
                            if res.text == "Ok!":
                                print("Upload Success!")
                            else:
                                print("Upload Failed ...")
                                print(res.text)
                        except requests.exceptions.ConnectionError:
                            print('No internet connection , upload failed ...')
                
                if len(idxSetOff) > 0:
                    indexPerson = list(idxSetOff)[0]
                    idxSet.remove(indexPerson)
                    index = indexPerson % 30
                    if attTrue[index] == attFalse[index]:
                        attTrue[index] -= 1
                    if writeStatus[index][attTrue[index]-1] == True:
                        countList += 1
                        sessID += 1
                        print(f'{sessID}|{indexPerson+1}|{genderID[index]}|{ageID[index]}|{durationDataID[index][attTrue[index]-1]}|{interestDataID[index][attTrue[index]-1]}|{date}|{timeID[index][0]}|{pic_name[index][0]}')
                        txt_file.write(f'{sessID},{indexPerson+1},{genderID[index]},{ageID[index]},{durationDataID[index][attTrue[index]-1]},{interestDataID[index][attTrue[index]-1]},{date},{timeID[index][0]},{pic_name[index][0]}\n')
                        payload["sess_id"] = sessID
                        payload["person_id"] = indexPerson+1
                        payload["gender"] = genderID[index]
                        payload["age"] = ageID[index]
                        payload["att_dur"] = durationDataID[index][attTrue[index]-1]
                        payload["interest"] = interestDataID[index][attTrue[index]-1]
                        payload["date"] = date
                        payload["time"] = timeID[index][0]
                        payload["cam"] = cam
                        payload["filename"] = pic_name[index][0]
                        payload["session"] = session
                        payload["robot_id"] = robotID
                        writeStatus[index][attTrue[index]-1] = False
                        try:
                            res = requests.post('https://ropi.web.id/api/pos.php', data=payload)
                            if res.text == "Ok!":
                                print("Upload Success!")
                            else:
                                print("Upload Failed ...")
                                print(res.text)
                        except requests.exceptions.ConnectionError:
                            print('No internet connection , upload failed ...')

        else:
            if noFace_status:
                print("No face detected")
                noFace_status = False

        durationMaster = round(durationMaster)
        sec_duration = durationMaster % 60
        min_duration = durationMaster // 60
        hour_duration = min_duration // 60
        seconds, minutes, hours = str(sec_duration).zfill(2), str(min_duration).zfill(2), str(hour_duration).zfill(2)

        fps = 1.0 // (startTime - prevStart_time)
        prevStart_time = startTime
        cv2.putText(frame, f'FPS: {str(fps)}', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("Frame", frame)
        
        
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif process2()=="tutup":
            break
        elif k == ord(' '):
            cv2.waitKey(-1)
            
    if len(idxSet) > 0:
        for indexPerson in list(idxSet):
            idxSet.remove(indexPerson)
            index = indexPerson % 30
            if attTrue[index] == attFalse[index]:
                attTrue[index] -= 1
            if writeStatus[index][attTrue[index]-1] == True:
                countList += 1
                sessID += 1
                print(f'{sessID}|{indexPerson+1}|{genderID[index]}|{ageID[index]}|{durationDataID[index][attTrue[index]-1]}|{interestDataID[index][attTrue[index]-1]}|{date}|{timeID[index][0]}|{pic_name[index][0]}')
                txt_file.write(f'{sessID},{indexPerson+1},{genderID[index]},{ageID[index]},{durationDataID[index][attTrue[index]-1]},{interestDataID[index][attTrue[index]-1]},{date},{timeID[index][0]},{pic_name[index][0]}\n')
                payload["sess_id"] = sessID
                payload["person_id"] = indexPerson+1
                payload["gender"] = genderID[index]
                payload["age"] = ageID[index]
                payload["att_dur"] = durationDataID[index][attTrue[index]-1]
                payload["interest"] = interestDataID[index][attTrue[index]-1]
                payload["date"] = date
                payload["time"] = timeID[index][0]
                payload["cam"] = cam
                payload["filename"] = pic_name[index][0]
                payload["session"] = session
                payload["robot_id"] = robotID
                writeStatus[index][attTrue[index]-1] = False
                try:
                    res = requests.post('https://ropi.web.id/api/pos.php', data=payload)
                    if res.text == "Ok!":
                        print("Upload Success!")
                    else:
                        print("Upload Failed ...")
                        print(res.text)
                except requests.exceptions.ConnectionError:
                    print('No internet connection , upload failed ...')

Process(target=process2).start()                    
Process(target=process1).start()

if len(csvList) > 0:
    newFile_path = f"{data_path}/data_{date}_{str(listDir_order+1).zfill(3)}.csv"
    newFile = open(newFile_path, 'a+')
    newFile.write('Session ID,Person ID,Gender,Age,Attention Duration,Interest,Date,Time,Files Name\n')

    for i in csvList:
        file = open(i, 'r')
        readData = file.readlines()
        for data in readData[1:]:
            newFile.write(data)
    print(f'Data saved in: {newFile_path}')
    print()
    print(f'Picture saved in {pic_path}/')
    print()
    
print(f"Program Running Duration: {hours}:{minutes}:{seconds}")

cv2.destroyAllWindows()
video.release()