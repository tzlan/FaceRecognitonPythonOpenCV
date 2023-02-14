import cv2
import face_recognition
import numpy as np
import pickle
from multiprocessing.pool import ThreadPool as Pool
import os

if not os.path.exists('pickle'):
    os.mkdir('pickle')

video_capture = cv2.VideoCapture(0)

known_face_encodings = {
    "Joan": "JOAN.jpeg",
    "Tal": "Tal.jpeg",
    "Raph": "raph.jpg",
    "Sahnip": "sahnip.jpg",
    "Mikael": "mika.jpg"
}


def load_image(name, img):
    pic = face_recognition.face_encodings(img)[0]
    with open(f"pickle/{name}.pickle", 'wb') as f:
        pickle.dump(pic, f, protocol=pickle.HIGHEST_PROTOCOL)


pool = Pool(2)

for name, file in known_face_encodings.items():
    loaded_img = face_recognition.load_image_file(file)
    pool.apply_async(load_image, (name, loaded_img,))

pool.close()
pool.join()

# retrieve from pickle
for name in list(known_face_encodings.keys()):
    filename = f"pickle/{name}.pickle"
    with open(filename, 'rb') as handle:
        known_face_encodings[name] = pickle.load(handle)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:

    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convertir l'image de la couleur BGR (qu'OpenCV utilise) à la couleur RGB (qu'utilise face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Ne traiter qu'une image sur deux de la vidéo pour gagner du temps
    if process_this_frame:
        # Trouver tous les visages et les encodages de visages dans l'image courante de la vidéo.
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Voir si le visage correspond au(x) visage(s) connu(s)
            matches = face_recognition.compare_faces(list(known_face_encodings.values()), face_encoding)
            name = "Inconnu"

            # # Si une correspondance a été trouvée dans known_face_encodings, utilisez simplement la première.
            # if True in matches :
            # first_match_index = matches.index(True)
            # name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(list(known_face_encodings.values()), face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = list(known_face_encodings.keys())[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        from random import randint

        cv2.rectangle(frame, (left, top), (right, bottom), (randint(0, 255), randint(0, 255), randint(0, 255)), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (randint(0, 255), randint(0, 255), randint(0, 255)),
                      cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # q pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ouverture de la camera
video_capture.release()
cv2.destroyAllWindows()
