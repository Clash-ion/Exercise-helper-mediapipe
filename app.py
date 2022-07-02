# Set Mode by camera or path to file
# For reading from video file simply put your videofile in same directory and then change MODE to './videofilename.mp4'
MODE = 'Camera' 

# Importing necessary libraries
import time
import cv2
from flask import Flask, render_template
from flask.wrappers import Response
import mediapipe as mp
import numpy as np
import os
import logging
logging.getLogger('werkzeug').disabled = True
# logging.getLogger('flask').disabled = True
os.environ['WERKZEUG_RUN_MAIN'] = 'true'

# A function to genertate a white image of size 512 * 512
def generate_empty_image():
    return np.ones(shape=(512, 512, 3), dtype=np.int16)

# A function to calculate the angle between two vectors
def angle_between(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return None
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# A function to convert list to np.array
def make_vector(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return np.array([x2 - x1, y2 - y1, z2 - z1])

# A function to convert radian to degree
def rad2deg(rad):
    return rad * 180 / np.pi

# A function to convert degree to radian
def deg2rad(deg):
    return deg * np.pi / 180

# A function to convery np.array to list
def np_to_list(np_array):
    return [x for x in np_array]


app = Flask(__name__)

@app.route('/')
def index():
    """Home page."""
    LOGGER.log(logging.WARNING, 'Home page accessed')
    return render_template('index.html')

class data:
    def __init__(self) -> None:
        self.angle = None
        self.reps = 0
        self.time = 0
        self.extras = ""

    def set_data(self, angle, reps):
        self.angle = angle
        self.reps = reps

    def set_time(self, time):
        self.time = time

    def set_extras(self, extras):
        self.extras = extras

    def get_angle(self):
        yield str(self.angle)

    def get_time(self):
        yield str(self.time)

    def get_reps(self):
        yield str(self.reps)

    def get_extras(self):
        yield str(self.extras)


global DATAOBJ
DATAOBJ = data()
global LOGGER
LOGGER = logging.getLogger(__name__)

def gen():

    previous_time = 0
    # creating our model to draw landmarks
    mpDraw = mp.solutions.drawing_utils
    # creating our model to detected our pose
    my_pose = mp.solutions.pose
    pose = my_pose.Pose(
        # model_complexity=2,
        # min_trackable_confidence=0.6,
        smooth_landmarks=True
    )

    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
    if MODE != 'Camera':
        cap = cv2.VideoCapture("./KneeBendVideo.mp4")

    """
    Local Variables to store few data
    """
    SMOOTH_ARR = []
    dataThreshold = 5
    threshold = 50
    start_timer = 0
    reps = 0
    flag = True

    while True:
        success, img = cap.read()
        # converting image to RGB from BGR because mediapipe only work on RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = pose.process(imgRGB)
        if result.pose_landmarks:
            mpDraw.draw_landmarks(
                img, result.pose_landmarks, my_pose.POSE_CONNECTIONS)
        data = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten(
        ) if result.pose_landmarks else np.zeros(33*4)
        data = data.reshape(33, 4)
        LEG_POINTS = [
            [23, 25, 27],
            [24, 26, 28]
        ]

        # Check for the visibility of the leg
        # further process is only donw when leg is visible

        visibilityflag = False
        v = []
        for leg in LEG_POINTS:
            tot = 0
            for j in leg:
                tot += data[j][3]
            v.append(tot/3)
        if v[0] > 0.6:
            v = []
            for i in LEG_POINTS[0]:
                v.append(np_to_list(data[i][:3]))
                visibilityflag = True

        elif v[1] > 0.6:
            v = []
            for i in LEG_POINTS[1]:
                v.append(np_to_list(data[i][:3]))
                visibilityflag = True
        else:
            visibilityflag = False
            DATAOBJ.set_extras("Leg not visible")

        if visibilityflag:
            try:
                angle = angle_between(
                    make_vector(v[0], v[1]),
                    make_vector(v[1], v[2]) * -1
                )
            except:
                angle = angle_between(np.array([0, 0, 0]), np.array([0, 0, 0]))
            #  if angle is less than 140 degrees, then we start timer and if timer is greater than 8 seconds, then we increase the reps by 1
            # leg is bent and previously it was straight



            if angle is not None:
                if len(SMOOTH_ARR) == dataThreshold:
                    # SMOOTH_ARR.append(angle)
                    if (np.absolute(np.median(SMOOTH_ARR)-rad2deg(angle)) < 3):
                        SMOOTH_ARR.pop(0)
                        SMOOTH_ARR.append(rad2deg(angle))
                        angle = deg2rad(np.median(SMOOTH_ARR))
                    # this threshold value can be tuned to remove sudden fluctutations. However lowering it to much will result in more false positives.
                    elif (np.absolute(np.median(SMOOTH_ARR)-rad2deg(angle)) > threshold): # should be tuned with any gradient techniques
                        angle = deg2rad(np.median(SMOOTH_ARR))
                    else:
                        SMOOTH_ARR.pop(0)
                        SMOOTH_ARR.append(rad2deg(angle))
                if len(SMOOTH_ARR) < dataThreshold:
                    if rad2deg(angle) < 140:  # should be tuned with any gradient techniques
                        angle = deg2rad(140)  # making that random noise at start of the model is not taken into account as that time mediapipe model will be trying to identify points so it may give wrong output
                                              # should be tuned with any gradient techniques
                    SMOOTH_ARR.append(rad2deg(angle))


            # save angle in a file
            LOGGER.log(logging.INFO,"Angle: " + str(angle))
            # for implementing continous time

            if angle is not None and flag:
                if start_timer == 0 and rad2deg(angle) < 140:
                    start_timer = time.time()
                elif start_timer != 0 and time.time() - start_timer >= 8 and rad2deg(angle) > 150:
                    # 8 sec over and angle is greater than 150 degrees. I choose 150 because it indicate that angle is moving higher that means leg is straightening.
                    flag = False
                elif start_timer != 0 and time.time() - start_timer < 8 and rad2deg(angle) > 140:
                    DATAOBJ.set_extras("KEEP YOUR KNEE BENT")
            if time.time() - start_timer > 10000:
                DATAOBJ.set_time(str(0))
            else:
                DATAOBJ.set_time(str(round(time.time() - start_timer,1)))
            if angle is not None and rad2deg(angle) < 140:
                DATAOBJ.set_extras("GOOD")

            if angle is not None and not flag and time.time() - start_timer >= 8:
                if rad2deg(angle) > 150:  # leg is straight and previously it was bent
                    reps += 1
                    flag = True
                    start_timer = 0

            if angle == None:  # if angle is None, then we reset the timer
                start_timer = 0
                degree = -1

            else:
                degree = rad2deg(angle)
            current_time = time.time()
            LOGGER.log(logging.INFO, f'fps :{1 / (current_time - previous_time)}')
            previous_time = current_time
            DATAOBJ.set_data(
                round(degree,4),
                reps
            )
        img = cv2.resize(img, (1068, 801))
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        key = cv2.waitKey(20)
        if key == 27:
            break


@app.route('/video_feed')
def video_feed():
    """Video stream route."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/data_angle')
def data_angle():
    """For getting the angle of the legs"""
    return Response(DATAOBJ.get_angle(),
                    mimetype='text')



@app.route('/data_reps')
def data_reps():
    """For getting the reps"""
    return Response(DATAOBJ.get_reps(),
                    mimetype='text')


@app.route('/data_time')
def data_time():
    """For data time i.e how much time since the user bent the leg or cross the thresshold"""
    return Response(DATAOBJ.get_time(),
                    mimetype='text')


@app.route('/data_extras')
def data_extras():
    """For extras like GOOD JOB, KEEP YOUR KNEE BENT, LEG NOT VISIBLE"""
    return Response(DATAOBJ.get_extras(),
                    mimetype='text')


if __name__ == "__main__":
    LOGGER.log(logging.WARNING, "Starting the application")
    LOGGER.log(logging.WARNING, "Application running at http://localhost:5000/")
    app.run()
