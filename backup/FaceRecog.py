from operator import index
import face_recognition
import cv2 as cv
import numpy as np
import os
 
path = 'images'
listOfKnownImages = []
listOfKnownPerson = []
imagesDir = os.listdir(path)

for img in imagesDir:
    currImg = cv.imread(f"{path}/{img}")
    listOfKnownImages.append(currImg)
    personName = os.path.splitext(img)[0]
    listOfKnownPerson.append(personName)

def encodeFaces(images):
    encodedList = []
    coloredImg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(coloredImg)[0]
    encodedList.append(encode)

    return encodedList

knownFaces = encodeFaces(listOfKnownImages)

Cam = cv.VideoCapture(1)

while True:
    success, img = Cam.read()

    if success == True:
        imgMod = cv.resize(img, (0,0), None, 0.25, 0.25)
        imgMod = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        currFaceLocFrame = face_recognition.face_locations(imgMod)

        encodeFaceFrame = face_recognition.face_encodings(imgMod, currFaceLocFrame)

        for encodedFace, faceLoc in zip(encodeFaceFrame, currFaceLocFrame):
            matchesFaces = face_recognition.compare_faces(knownFaces, encodedFace)
            faceDistance = face_recognition.face_distance(knownFaces, encodedFace)

            print(faceDistance)

            index = np.argmin(faceDistance)

            if matchesFaces[index]:

                name = listOfKnownPerson[index].upper()

                print(name)

                y1, x2, y2, x1 = faceLoc

                cv.rectangle(img, (x1,y1), (x2,y2), (255, 97, 98), 2)
                cv.rectangle(img, (x1,y2-35), (x2,y2), (255, 97, 98), cv.FILLED)
                cv.putText(img, name, (x1+6, y2-6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            cv.imshow("Face Recognition", img)  
        
        if cv.waitKey(1) & 0xFF == ord('o'):
            break

             