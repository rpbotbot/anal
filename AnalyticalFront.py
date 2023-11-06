import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import time
import numpy as np
from Tracker.centroidtracker import CentroidTracker
import xlsxwriter
import os
import random

faceProto = "face_deploy.prototxt"
faceModel = "face_net.caffemodel"
faceNet = cv2.dnn.readNetFromCaffe(faceProto, faceModel)
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
ageNet = cv2.dnn.readNetFromCaffe(ageProto, ageModel)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

genderList = ['Male', 'Female']
# ageList = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']
ageList = ['(0 - 5)', '(6 - 11)', '(12 - 16)', '(17 - 21)', '(22 - 30)', '(31 - 40)', '(41 - 55)', '(56 - 100)']

cam = 0
video = cv2.VideoCapture(cam)
video.set(3, 960)
video.set(4, 540)
if cam == 0:
    cam = 1
if cam == 1:
    cam = 2
detector = FaceMeshDetector()
ct = CentroidTracker()

date = time.strftime('%Y-%m-%d')
status = 'Front'

xlsx_path = f"C:/Users/NUC 11PAHi5/Pictures/Analytical/Dataset/Analytical_Person/{date}/{status}/Data"
pic_path = f"C:/Users/NUC 11PAHi5/Pictures/Analytical/Dataset/Analytical_Person/{date}/{status}/Picture"

isDir_xlsx = os.path.isdir(xlsx_path)
isDir_pic = os.path.isdir(pic_path)
if not isDir_xlsx:
    os.makedirs(xlsx_path)
if not isDir_pic:
    os.makedirs(pic_path)

length_data = len(os.listdir(f"C:/Users/NUC 11PAHi5/Pictures/Analytical/Dataset/Analytical_Person/{date}/{status}/Data"))

workbook = xlsxwriter.Workbook(f"{xlsx_path}/data_{date}_{length_data+1}.xlsx")
worksheet = workbook.add_worksheet()

row = 0
column = 0

idList = [127, 33, 6, 359, 356]
idList2 = [127, 152, 10, 356, 151]

fontStyle = cv2.FONT_HERSHEY_DUPLEX
fontColor = (0, 0, 0)
fontSize = 0.45

confidence = 0.5

startDuration = 0
prev_start_time = 0
index_request = 0
startProgram = True

idxList = []
idxStatus = []
idxSet = set()

durationID = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
start_durationID = [
    [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], 
    [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], 
    [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], 
    ]
durationDataID = [
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], 
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], 
    ]
totalDurationID = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
    
attTrue = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    ]
attFalse = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    ]

genderPredID = [
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    ]
agePredID = [
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    ]

durationPred = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
start_durationPred = [
    [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], 
    [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], 
    [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], [0, True], 
    ]
    
interest = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
interestDataID = [
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], 
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], 
    ]

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

timeID = [
    ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], 
    ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], 
    ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], 
    ]
timeDataID = [
    [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], 
    [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""],
    ]

pic_name = [
    ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], 
    ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], 
    ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], ["", True], 
    ]
pic_nameDataID = [
    [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], 
    [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""],
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

format = workbook.add_format()
format.set_align('center')

worksheet.set_column(0, 0, 10)
worksheet.set_column(4, 4, 17)
worksheet.set_column(5, 6, 10)
worksheet.set_column(8, 8, 25)
worksheet.set_column(0, 8, cell_format=format)

worksheet.write(row, column, 'Session ID')
worksheet.write(row, column+1, 'Person ID')
worksheet.write(row, column+2, 'Gender')
worksheet.write(row, column+3, 'Age')
worksheet.write(row, column+4, 'Attention Duration')
worksheet.write(row, column+5, 'Interest')
worksheet.write(row, column+6, 'Date')
worksheet.write(row, column+7, 'Time')
worksheet.write(row, column+8, 'Files Name')

while True:
    start_time = time.time()
    count = round(start_time, 2)
    if startProgram:
        startDuration = count
        startProgram = False
    duration = round((count - startDuration), 2)
    ret, frame = video.read()
    if not ret:
        print("Camera disconnected")
        break
    H, W = frame.shape[:2]
    # h_resize, w_resize = round(H*0.5), round(W*0.5)
    # frame = cv2.resize(frame, (w_resize, h_resize))
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    detect_object = detections[0,0,:,1]
    length_of_object = len(detect_object)
    rects = []
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > confidence:
            new_image = frame.copy()
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))
            
    objects = ct.update(rects)
    if len(objects) > 0:
        for (objectID, centroid) in objects.items():
            index = objectID % 30
            indexPerson = objectID+1
            str_indexPerson = str(indexPerson)
            index_clear = index + 15
            if index_clear > 29:
                index_clear = index_clear - 30
            start_durationID[index_clear] = [0, True]
            start_durationPred[index_clear] = [0, True]
            genderPredID[index_clear] = []
            agePredID[index_clear] = []
            attTrue[index_clear] = 1
            attFalse[index_clear] = 1
            durationID[index_clear] = 0
            durationDataID[index_clear] = [0]
            timeID[index_clear] = ["", True]
            timeDataID[index_clear] = [""]
            interest[index_clear] = 0
            interestDataID[index_clear] = [0]
            pic_name[index_clear] = ["", True]
            pic_nameDataID[index_clear] = [""]
            startX, startY, endX, endY = centroid[2], centroid[3], centroid[4], centroid[5]
            h_rect = endX - startX
            ratioFont = h_rect/h_rect*0.5
            text = "Person {}".format(str_indexPerson)
            cv2.putText(frame, text, (centroid[2], centroid[3] - 10), fontStyle, ratioFont, (0, 255, 0), 1)
            boxSize = (endX - startX)*(endY - startY)

            # DEFINE ROI
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
            image, faces = detector.findFaceMesh(ROI, draw=False)
            if faces:
                # DEFINE DOTS
                face = faces[0]
                leftSideA, leftSideB, rightSideA, rightSideB, center = face[356], face[359], face[127], face[33], face[6]
                upperSideA, upperSideB, bottomSideA, bottomSideB = face[10], face[151], face[200], face[152]
                areaUpper = round(detector.findDistance(upperSideA, upperSideB)[0])
                areaBottom = round(detector.findDistance(bottomSideA, bottomSideB)[0])
                areaLeft = round(detector.findDistance(leftSideB, center)[0])
                areaRight = round(detector.findDistance(rightSideB, center)[0])

                # CALCULATE THE DEVIATIONS
                deviationHorizontal = abs(areaLeft - areaRight)
                deviationVertical = abs(areaUpper - areaBottom)
                # print(f'Box Size: {boxSize}, Deviation Horizontal: {deviationHorizontal}, Deviation Vertical: {deviationVertical}')
                if start_durationPred[index][1]:
                    start_durationPred[index][0] = duration
                    start_durationPred[index][1] = False
                durationPred[index] = round((duration - start_durationPred[index][0]), 2)
                if durationPred[index] < 5:
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
                    sub_frame = frame[startY_stats:endY_stats, startX_stats:endX_stats]
                    white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                    res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
                    frame[startY_stats:endY_stats, startX_stats:endX_stats] = res
                    
                else:
                    genderID[index] = max(set(genderPredID[index]), key=genderPredID[index].count)
                    ageID[index] = max(set(agePredID[index]), key=agePredID[index].count)
                
                if ageID[index] == '(0 - 5)' or ageID[index] == '(6 - 11)':
                    ageID[index] = 'Kids'
                if ageID[index] == '(12 - 16)' or ageID[index] == '(17 - 21)':
                    ageID[index] = 'Young'
                if ageID[index] == '(22 - 30)' or ageID[index] == '(31 - 40)':
                    ageID[index] = 'Adult'
                if ageID[index] == '(41 - 55)' or ageID[index] ==  '(56 - 100)':
                    ageID[index] = 'Senior'
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
                        if start_durationID[index][1]:
                            start_durationID[index][0] = duration
                            start_durationID[index][1] = False
                        durationID[index] = round((duration - start_durationID[index][0]), 2)
                        str_att = str(durationID[index])
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
                        cv2.putText(frame, str_att, (startX+2, endY+45), fontStyle, fontSize, fontColor, 1)            
                        durationDataID[index][attTrue[index]-1] = durationID[index]
                        timeDataID[index][attTrue[index]-1] = timeID[index][0]
                        totalDurationID[index] = sum(durationDataID[index])
                        if durationID[index] >= 0.5:
                            if pic_name[index][1]:
                                pic_name[index][0] = f"{length_data+1}_{date}_id{indexPerson}_att{attTrue[index]}.jpg"
                                pic_nameDataID[index][attTrue[index]-1] = pic_name[index][0]
                                cv2.imwrite(f"{pic_path}/{pic_name[index][0]}", ROI)
                                print(f"Picture saved as {pic_name[index][0]}")
                            if attFalse[index] == attTrue[index]:
                                attFalse[index] += 1
                                idxList.append([indexPerson, attTrue[index], index])
                                idxSet.add(indexPerson - 1)
                                idxStatus.append(True)
                            pic_name[index][1] = False
                            sessID = idxList.index([indexPerson, attTrue[index], index])
                            # print(f"Index: {sessID}, Person: {indexPerson}, Duration: {durationID[index]}, Attention: {attTrue[index]}")
                            # print(idxList)

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
                        start_durationID[index][1] = True
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
                        if start_durationID[index][1]:
                            start_durationID[index][0] = duration
                            start_durationID[index][1] = False
                        durationID[index] = round((duration - start_durationID[index][0]), 2)
                        str_att = str(durationID[index])
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
                        cv2.putText(frame, str_att, (startX+2, endY+45), fontStyle, fontSize, fontColor, 1)            
                        durationDataID[index][attTrue[index]-1] = durationID[index]
                        timeDataID[index][attTrue[index]-1] = timeID[index][0]
                        totalDurationID[index] = sum(durationDataID[index])
                        if durationID[index] >= 0.5:
                            if pic_name[index][1]:
                                pic_name[index][0] = f"{length_data+1}_{date}_id{indexPerson}_att{attTrue[index]}.jpg"
                                pic_nameDataID[index][attTrue[index]-1] = pic_name[index][0]
                                cv2.imwrite(f"{pic_path}/{pic_name[index][0]}", ROI)
                                print(f"Picture saved as {pic_name[index][0]}")
                            if attFalse[index] == attTrue[index]:
                                attFalse[index] += 1
                                idxList.append([indexPerson, attTrue[index], index])
                                idxSet.add(indexPerson - 1)
                                idxStatus.append(True)
                            pic_name[index][1] = False
                            sessID = idxList.index([indexPerson, attTrue[index], index])
                            # print(f"Index: {sessID}, Person: {indexPerson}, Duration: {durationID[index]}, Attention: {attTrue[index]}")
                            # print(idxList)

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
                        start_durationID[index][1] = True
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
                        if start_durationID[index][1]:
                            start_durationID[index][0] = duration
                            start_durationID[index][1] = False
                        durationID[index] = round((duration - start_durationID[index][0]), 2)
                        str_att = str(durationID[index])
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
                        cv2.putText(frame, str_att, (startX+2, endY+45), fontStyle, fontSize, fontColor, 1)            
                        durationDataID[index][attTrue[index]-1] = durationID[index]
                        timeDataID[index][attTrue[index]-1] = timeID[index][0]
                        totalDurationID[index] = sum(durationDataID[index])
                        if durationID[index] >= 0.5:
                            if pic_name[index][1]:
                                pic_name[index][0] = f"{length_data+1}_{date}_id{indexPerson}_att{attTrue[index]}.jpg"
                                pic_nameDataID[index][attTrue[index]-1] = pic_name[index][0]
                                cv2.imwrite(f"{pic_path}/{pic_name[index][0]}", ROI)
                                print(f"Picture saved as {pic_name[index][0]}")
                            if attFalse[index] == attTrue[index]:
                                attFalse[index] += 1
                                idxList.append([indexPerson, attTrue[index], index])
                                idxSet.add(indexPerson - 1)
                                idxStatus.append(True)
                            pic_name[index][1] = False
                            sessID = idxList.index([indexPerson, attTrue[index], index])
                            # print(f"Index: {sessID}, Person: {indexPerson}, Duration: {durationID[index]}, Attention: {attTrue[index]}")
                            # print(idxList)

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
                        start_durationID[index][1] = True
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
                        if start_durationID[index][1]:
                            start_durationID[index][0] = duration
                            start_durationID[index][1] = False
                        durationID[index] = round((duration - start_durationID[index][0]), 2)
                        str_att = str(durationID[index])
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
                        cv2.putText(frame, str_att, (startX+2, endY+45), fontStyle, fontSize, fontColor, 1)            
                        durationDataID[index][attTrue[index]-1] = durationID[index]
                        timeDataID[index][attTrue[index]-1] = timeID[index][0]
                        totalDurationID[index] = sum(durationDataID[index])
                        if durationID[index] >= 0.5:
                            if pic_name[index][1]:
                                pic_name[index][0] = f"{length_data+1}_{date}_id{indexPerson}_att{attTrue[index]}.jpg"
                                pic_nameDataID[index][attTrue[index]-1] = pic_name[index][0]
                                cv2.imwrite(f"{pic_path}/{pic_name[index][0]}", ROI)
                                print(f"Picture saved as {pic_name[index][0]}")
                            if attFalse[index] == attTrue[index]:
                                attFalse[index] += 1
                                idxList.append([indexPerson, attTrue[index], index])
                                idxSet.add(indexPerson - 1)
                                idxStatus.append(True)
                            pic_name[index][1] = False
                            sessID = idxList.index([indexPerson, attTrue[index], index])
                            # print(f"Index: {sessID}, Person: {indexPerson}, Duration: {durationID[index]}, Attention: {attTrue[index]}")
                            # print(idxList)

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
                        start_durationID[index][1] = True
                        sub_frame = frame[startY_stats:endY_stats, startX_stats:endX_stats]
                        white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                        res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
                        frame[startY_stats:endY_stats, startX_stats:endX_stats] = res
                        cv2.putText(frame, f'Gender: {finalGender}', (startX+2, endY+15), fontStyle, fontSize, fontColor, 1)
                        cv2.putText(frame, f'Age: {finalAge}', (startX+2, endY+30), fontStyle, fontSize, fontColor, 1)
                             
            else:
                print("Face is not accurate")
                timeID[index][1] = True
                start_durationID[index][1] = True
                start_durationPred[index][1] = True
                genderPredID[index] = []
                agePredID[index] = []

            if len(durationDataID[index]) == attTrue[index]-1:
                durationDataID[index].append(0)
            if len(timeDataID[index]) == attTrue[index]-1:
                timeDataID[index].append("")
            if len(pic_nameDataID[index]) == attTrue[index]-1:
                pic_nameDataID[index].append("")
            if len(interestDataID[index]) == attTrue[index]-1:
                interestDataID[index].append(0)

            if durationID[index] >= 0.5 and durationID[index] < 2.5:
                interest[index] = 0
            if durationID[index] >= 2.5 and durationID[index] < 5:
                interest[index] = 1
            if durationID[index] >= 5:
                interest[index] = 2
            interestDataID[index][attTrue[index]-1] = interest[index]

            if durationID[index] > 0.5:
                worksheet.write(sessID+1, column, sessID+1)
                worksheet.write(sessID+1, column+1, indexPerson)
                worksheet.write(sessID+1, column+2, genderID[index])
                worksheet.write(sessID+1, column+3, ageID[index])
                worksheet.write(sessID+1, column+4, durationID[index])
                worksheet.write(sessID+1, column+5, interest[index])       
                worksheet.write(sessID+1, column+6, date)
                worksheet.write(sessID+1, column+7, timeID[index][0])         
                worksheet.write(sessID+1, column+8, pic_name[index][0])        

    else:
        print("No Face")

    duration = round(duration)
    sec_duration = duration % 60
    min_duration = duration // 60
    hour_duration = min_duration // 60
    seconds, minutes, hours = str(sec_duration).zfill(2), str(min_duration).zfill(2), str(hour_duration).zfill(2)

    fps = 1.0 // (start_time - prev_start_time)
    prev_start_time = start_time
    cv2.putText(frame, f'FPS: {str(fps)}', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
    if k == ord(' '):
        cv2.waitKey(-1)

numpy_idxStatus = np.array(idxStatus)
idxTrue = np.where(numpy_idxStatus == True)

print(idxList)
print()
print(idxStatus)
print()
print(durationDataID)
print()
print(timeDataID)
print()
print(pic_nameDataID)
print()

workbook.close()
print(f"Program Running Duration: {hours}:{minutes}:{seconds}")
print(f'Data saved in {xlsx_path}/data_{date}_{length_data+1}.xlsx')
print(f'Picture saved in {pic_path}/')
cv2.destroyAllWindows()
video.release