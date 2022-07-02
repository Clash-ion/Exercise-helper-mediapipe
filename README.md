# Mediapipe Exercise Helper

## **Abstract:**

The detector is inspired by the lightweight
[BlazeFace](https://arxiv.org/abs/1907.05047) model, used in [MediaPipe
Face
Detection](https://google.github.io/mediapipe/solutions/face_detection.html),
as a proxy for a person detector. It explicitly predicts two additional
virtual key points that firmly describe the human body centre, rotation
and scale as a circle. The landmark model in MediaPipe Pose predicts the
location of 33 pose landmarks.

## **Process:**

Since the Mediapipe API gives 33 landmarks, we will use these landmarks
to get the details about the position of each body part. We are looking
for landmarks of the hip, knee and ankle as per the requirement. Now
since we have two legs and this exercise will mostly be done sideways
i.e. the camera will be facing the side of the person so we have to
choose which leg is in front and which is in the back. Further down the
line, I have created logic which tells the program to increase the
counter if the user has completed their rep.

## **Testing:**

On the given testing data the program seems to be working fine. Although
sometimes it starts the timer off time due to noisy data points provided
by the Mediapipe API. For this, we can use exponential smoothing.
Furthermore, we can remove the effect of redundant frames with the help
of exponential smoothing. If the change is greater than the threshold
the program will retain the previous value but if the change is small
the program will first smooth it to then make sure that the angle is
increasing gradually because the leg, in this case, will be moving
slowly.

## **Steps to RUN:**

1.  RUN the following command in Command-Line

    `pip install -r requirements.txt`

2.  `python app.py`

3.  Now you can visit
    [[http://localhost:5000]{.ul}](http://localhost:5000) to access
    the result.

> Note: It may take a few seconds for mediapipe to start itself as it
> needs to start its underlying tflite API.
