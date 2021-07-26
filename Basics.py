import cv2
import numpy as np
import face_recognition

# The basic structure of face recognition test model which detects the exact faces with the test faces.
# @author ARINC

imgA = face_recognition.load_image_file('imageBasics/Chandler bing.jpg')
imgA = cv2.cvtColor(imgA,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('imageBasics/Monica geller.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgA)[0]
encodeElon = face_recognition.face_encodings(imgA)[0]
cv2.rectangle(imgA,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)


cv2.imshow('Chandler',imgA)
cv2.imshow('Monica',imgTest)
cv2.waitKey(0)