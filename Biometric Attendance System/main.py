import face_recognition
import  cv2
import numpy as np
import csv
from datetime import datetime



video_capture = cv2.VideoCapture(0)
#load known faces

my_img = face_recognition.load_image_file("faces/me.jpg")
me_encoding = face_recognition.face_encodings(my_img)[0]


anuj_img = face_recognition.load_image_file("faces/anuj.jpg")
anuj_encoding = face_recognition.face_encodings(anuj_img)[0]

shubham_img = face_recognition.load_image_file("faces/shubham.jpg")
shubham_encoding = face_recognition.face_encodings(shubham_img)[0]

kajal_img = face_recognition.load_image_file("faces/kajal.jpg")
kajal_encoding = face_recognition.face_encodings(kajal_img)[0]

harshada_img = face_recognition.load_image_file("faces/harshada.jpg")
harshada_encoding = face_recognition.face_encodings(harshada_img)[0]

yashika_img = face_recognition.load_image_file("faces/yashika.jpg")
yashika_encoding = face_recognition.face_encodings(yashika_img)[0]

rohit_img = face_recognition.load_image_file("faces/rohit.jpg")
rohit_encoding = face_recognition.face_encodings(rohit_img)[0]

pankaj_img = face_recognition.load_image_file("faces/pankaj.jpg")
pankaj_encoding = face_recognition.face_encodings(pankaj_img)[0]

pramila_img = face_recognition.load_image_file("faces/pramila.jpg")
pramila_encoding = face_recognition.face_encodings(pankaj_img)[0]


known_face_encodings = [me_encoding,yashika_encoding,anuj_encoding,kajal_encoding,shubham_encoding,harshada_encoding,rohit_encoding,pankaj_encoding,pramila_encoding]
known_face_names =["Smaran", "Yashika","Anuj","kajal","shubham","Harshada","Rohit","Pankaj","Pramila"]

#List of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# Get the current date and time

now= datetime.now()
current_date =now.strftime("%y-%m-%d")

f = open(f"{current_date}.csv","w+", newline= "")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx= 0.25,fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    #Recognise faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
        face_distance=  face_recognition.face_distance(known_face_encodings,face_encoding)

        best_match_index = np.argmin(face_distance)


        if matches[best_match_index]:
            Name = known_face_names[best_match_index]

        if  Name in known_face_names:
            font =cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText =(10,100)
            fontScale=1.2
            fontColor = (255,255,255)
            thickness=2
            lineType=2

            cv2.putText(frame, Name + " present ", bottomLeftCornerOfText,font,fontScale,fontColor,thickness,lineType)

            if Name in students:
                students.remove(Name)
                current_time=now.strftime("%H-%M-%S")
                lnwriter.writerow([Name,current_time])


    cv2.imshow("Attendance",frame )
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()
f.close