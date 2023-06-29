import datetime as datetime
import face_recognition
import cv2
import datetime

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:

        time=str(datetime.datetime.now())

        img_name = "sample.jpeg".format(img_counter)
        cv2.imwrite("./dataset/"+img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        break
cam.release()

cv2.destroyAllWindows()

biden_image = face_recognition.load_image_file("./dataset/biden.jpg")
obama_image = face_recognition.load_image_file("./dataset/obama.jpg")
deepak_image = face_recognition.load_image_file("./dataset/deepak.jpeg")
akhil_image = face_recognition.load_image_file("./dataset/akhil.jpeg")
dona_image = face_recognition.load_image_file("./dataset/dona.jpeg")
delna_image = face_recognition.load_image_file("./dataset/delna.jpeg")

unknown_image = face_recognition.load_image_file("./dataset/"+img_name)

try:
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    deepak_face_encoding = face_recognition.face_encodings(deepak_image)[0]
    akhil_face_encoding = face_recognition.face_encodings(akhil_image)[0]
    akhil_face_encoding = face_recognition.face_encodings(akhil_image)[0]
    dona_face_encoding = face_recognition.face_encodings(dona_image)[0]
    delna_face_encoding = face_recognition.face_encodings(delna_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    biden_face_encoding,
    obama_face_encoding,
    deepak_face_encoding,
    akhil_face_encoding,
    dona_face_encoding,
    delna_face_encoding
]

results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("Is the unknown face a picture of Biden? {}".format(results[0]))
print("Is the unknown face a picture of Obama? {}".format(results[1]))
print("Is the unknown face a picture of Deepak? {}".format(results[2]))
print("Is the unknown face a picture of Akhil? {}".format(results[3]))
print("Is the unknown face a picture of Dona? {}".format(results[4]))
print("Is the unknown face a picture of Delna? {}".format(results[5]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))



