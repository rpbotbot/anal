import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import time
import numpy as np
from Tracker.centroidtracker import CentroidTracker
import xlsxwriter
import os
import random
import requests
# from deepface import DeepFace

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

detector = FaceMeshDetector()
ct = CentroidTracker()

date = time.strftime('%Y-%m-%d')

xlsx_path = f"C:/Users/NUC 11PAHi5/Pictures/Analytical/Dataset/Analytical_Person/{date}/Front/Data"
pic_path = f"C:/Users/NUC 11PAHi5/Pictures/Analytical/Dataset/Analytical_Person/{date}/Front/Picture"

# csv_path = f"C:/Users/whyag/Documents/Wahyu_Agung/Dataset/Analytical_Person/{date}/Rear/Data"
# pic_path = f"C:/Users/whyag/Documents/Wahyu_Agung/Dataset/Analytical_Person/{date}/Rear/Picture"
isDir_xlsx = os.path.isdir(xlsx_path)
isDir_pic = os.path.isdir(pic_path)
if not isDir_xlsx:
    os.makedirs(xlsx_path)
if not isDir_pic:
    os.makedirs(pic_path)

length_data = len(os.listdir(f"C:/Users/NUC 11PAHi5/Pictures/Analytical/Dataset/Analytical_Person/{date}/Front/Data/"))

workbook = xlsxwriter.Workbook(f"{xlsx_path}/data_{date}_{length_data+1}.xlsx")
worksheet = workbook.add_worksheet()

row = 0
column = 0

idList = [127, 33, 6, 359, 356]
idList2 = [127, 152, 10, 356]

fontStyle = cv2.FONT_HERSHEY_DUPLEX
fontColor = (0, 0, 0)
fontSize = 0.45

confidence = 0.5

startDuration = 0

take_picture = True
start = True

durationID = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
totalDurationID = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
durationDataID = [
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], 
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], 
    ]
    
gazeT = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
gazeF = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]

start_durationID = [
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    ]

genderPredID = [
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    ]
agePredID = [
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    ]
emotionPredID = [
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    ]

durationPred = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
    
start_durationPred = [
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    ]

times = [
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
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

emotionID = [
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "", 
    ]

pic_name = [
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    ]

statusUpload = [
    False, False, False, False, False, False,
    False, False, False, False, False, False,
    False, False, False, False, False, False,
    False, False, False, False, False, False,
    False, False, False, False, False, False,
    ]

sess_id = ''.join(random.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') for i in range(16))

payload = {
  'person_id': 0, # integer
  'gender': '', # string
  'age': '', # string
  'att_dur': 0.0, # float
  'att_times': 0, # integer
  'date': "YYYY-MM-DD", # string
  'time': "hh:mm:ss", # string
  'cam': 0, # integer
  'files': "xxx.jpg", # string
  'sess_id': "xxxxxxxxxxxxxxxx", # string
}

format = workbook.add_format()
format.set_align('center')

worksheet.set_column(3, 4, 17)
worksheet.set_column(5, 6, 10)
worksheet.set_column(7, 7, 20)
worksheet.set_column(0, 7, cell_format=format)

while True:
    start_time = time.time()
    count = round(time.time(), 2)
    if start:
        startDuration = count
        start = False
    duration = round((count - startDuration), 2)
    ret, frame = video.read()
    if not ret:
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
            statusUpload[index] = True
            times[index].append(time.strftime('%T'))
            index_clear = index + 15
            if index_clear > 29:
                index_clear = index_clear - 30
            if statusUpload[index_clear]:
                payload["person_id"] = indexPerson - 15
                payload["gender"] = genderID[index_clear]
                payload["age"] = ageID[index_clear]
                payload["att_dur"] = totalDurationID[index_clear]
                payload["att_times"] = gazeT[index_clear] + 1
                payload["date"] = date
                payload["time"] = times[index_clear][0]
                payload["files"] = pic_name[index_clear]
                payload["cam"] = cam
                payload["sess_id"] = sess_id
                res = requests.post('https://ropi.web.id/api/pos.php', data=payload)
                print(res.text)
                if res.text == "Ok!":
                    print("Upload Success")
                else:
                    print("Upload Failed")
                statusUpload[index_clear] = False

            start_durationID[index_clear] = []
            start_durationPred[index_clear] = []
            genderPredID[index_clear] = []
            agePredID[index_clear] = []
            genderID[index_clear] = ""
            ageID[index_clear] = ""
            gazeT[index_clear] = 0
            gazeF[index_clear] = 0
            times[index_clear] = []
            durationDataID[index_clear] = [0]
            pic_name[index_clear] = ""
            startX, startY, endX, endY = centroid[2], centroid[3], centroid[4], centroid[5]
            h_rect = endX - startX
            ratioFont = h_rect/h_rect*0.5
            text = "Person {}".format(str_indexPerson)
            cv2.putText(frame, text, (centroid[2], centroid[3] - 10), fontStyle, ratioFont, (0, 255, 0), 1)
            boxSize = (endX-startX)*(endY-startY)
            biasX = round((endX-startX)*0.3)
            biasY = round((endY-startY)*0.3)
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
                face = faces[0]
                leftSideA, leftSideB, rightSideA, rightSideB, center = face[356], face[359], face[127], face[33], face[6]
                areaLeft = round(detector.findDistance(leftSideB, center)[0])
                areaRight = round(detector.findDistance(rightSideB, center)[0])
                # print(f'Left: {lengthLeft}, Right: {lengthRight}')
                bias = abs(areaLeft - areaRight)
                start_durationPred[index].append(duration)
                durationPred[index] = round((duration - start_durationPred[index][0]), 2) 
                if durationPred[index] < 5:
                    blob = cv2.dnn.blobFromImage(ROI, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    genderNet.setInput(blob)
                    ageNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]
                    agePreds = ageNet.forward()
                    age = ageList[agePreds[0].argmax()]
                    # analyze = DeepFace.analyze(ROI, actions=['emotion'], detector_backend=backends[4])
                    # print(analyze)
                    # emotion = analyze['dominant_emotion']
                    genderPredID[index].append(gender)
                    agePredID[index].append(age)
                    genderID[index] = max(set(genderPredID[index]), key=genderPredID[index].count)
                    ageID[index] = max(set(agePredID[index]), key=agePredID[index].count)
                    # emotionPredID[index].append(emotion)
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
                    # emotionID[index] = max(set(emotionPredID[index]), key=emotionPredID[index].count)
                
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
                if bias < 15:
                    start_durationID[index].append(duration)
                    durationID[index] = round((duration - start_durationID[index][0]), 2) 
                    # durationDataID[index] = durationID[index]
                    str_gaze = str(durationID[index])
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
                    # cv2.putText(frame, f'Emotion: {emotionID[index]}', (startX+2, endY+45), fontStyle, fontSize, fontColor, 1)
                    cv2.putText(frame, str_gaze, (startX+2, endY+45), fontStyle, fontSize, fontColor, 1)
                    if durationID[index] >= 2:
                        if gazeF[index] == gazeT[index]:
                            gazeF[index] += 1
                    
                    print(f"Person: {objectID+1}, Duration: {durationID[index]}, Attention: {gazeT[index]+1}")
                    durationDataID[index][gazeT[index]] = durationID[index]
                    totalDurationID[index] = sum(durationDataID[index])

                    if totalDurationID[index] >= 5:
                        if take_picture:
                            pic_name[index] = f"{length_data+1}_{date}_id{objectID+1}.jpg"
                            cv2.imwrite(f"{pic_path}/{pic_name[index]}", ROI)
                            print("Picture taken")
                        take_picture = False
                        # cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), int(round(H/320)))
                                 
                    else:
                        take_picture = True
                        pic_name[index] = ''
                        
                else:
                    if gazeT[index] != gazeF[index]:
                        gazeT[index] += 1
                    startX_stats, endX_stats, startY_stats, endY_stats = startX, startX + 120, endY - 1, endY + 50
                    if startX_stats < 0:
                        startX_stats = 0
                    if startY_stats < 0:
                        startY_stats = 0
                    if endX_stats > W:
                        endX_stats = W
                    if endY_stats > H:
                        endY_stats = H
                    start_durationID[index] = []
                    sub_frame = frame[startY_stats:endY_stats, startX_stats:endX_stats]
                    white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                    res = cv2.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
                    frame[startY_stats:endY_stats, startX_stats:endX_stats] = res
                    cv2.putText(frame, f'Gender: {finalGender}', (startX+2, endY+15), fontStyle, fontSize, fontColor, 1)
                    cv2.putText(frame, f'Age: {finalAge}', (startX+2, endY+30), fontStyle, fontSize, fontColor, 1)
                    # cv2.putText(frame, f'Emotion: {emotionID[index]}', (startX+2, endY+45), fontStyle, fontSize, fontColor, 1)
            
            else:
                print("Face is not accurate")
                start_durationID[index] = []
                start_durationPred[index] = []
                genderPredID[index] = []
                agePredID[index] = []
                # emotionPredID[index] = []
            
            if len(durationDataID[index]) == gazeT[index]:
                durationDataID[index].append(0)

            worksheet.write(row, column, 'Person ID')
            worksheet.write(row, column+1, 'Gender')
            worksheet.write(row, column+2, 'Age')
            worksheet.write(row, column+3, 'Attention Duration')
            worksheet.write(row, column+4, 'Attention Times')
            worksheet.write(row, column+5, 'Date')
            worksheet.write(row, column+6, 'Time')
            worksheet.write(row, column+7, 'Picture Name')

            worksheet.write(objectID+1, column, indexPerson)
            worksheet.write(objectID+1, column+1, genderID[index])
            worksheet.write(objectID+1, column+2, ageID[index])
            worksheet.write(objectID+1, column+3, totalDurationID[index])
            worksheet.write(objectID+1, column+4, gazeT[index]+1)
            worksheet.write(objectID+1, column+5, date)
            worksheet.write(objectID+1, column+6, times[index][0])
            worksheet.write(objectID+1, column+7, pic_name[index])

    else:
        print("No Face")

    duration = round(duration)
    # print(duration)

    fps = str(1.0 // (time.time() - start_time))
    cv2.putText(frame, fps, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

print("Program Running Duration:", duration)
workbook.close()
print(f'Data saved in {xlsx_path}/data_{date}_{length_data+1}.xlsx')
print(f'Picture saved in {pic_path}/')
cv2.destroyAllWindows()
video.release