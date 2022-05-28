from flask import Flask, render_template, Response,url_for,redirect,request
import cv2
import face_recognition
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

fetch_name=""

# Load a sample picture and learn how to recognize it.
elon_image = face_recognition.load_image_file("photos/elon.jpg")
elon_face_encoding = face_recognition.face_encodings(elon_image)[0]

# Load a second sample picture and learn how to recognize it.
bill_image = face_recognition.load_image_file("photos/bill.jpg")
bill_face_encoding = face_recognition.face_encodings(bill_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    elon_face_encoding,
    bill_face_encoding
]
known_face_names = [
    "Elon",
    "Bill"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


def generate_frames():
    while True:
        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
                # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                global fetch_name
                fetch_name=name
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/success/<string:urname>')
def success(urname):
    res=urname
    return render_template('success.html',result=res)\

@app.route('/fail/<string:urname>')
def fail(urname):
    res = urname
    return render_template("fail.html",result=res)






@app.route('/submit',methods=['POST','GET'])
def submit():
    get_name=""
    if request.method=='POST':
        get_name=request.form['yourname']
    res=''

    if get_name==fetch_name:
        res='success'
    else:
        res="fail"
    return redirect(url_for(res,urname=get_name))

if __name__ == "__main__":
    app.run(debug=True)