# import the necessary packages
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

from DLLIVEPREDICT_CLASS import DLLivePredict_Class

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)
  
@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")
    
def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock
    
    live=DLLivePredict_Class()
    live.loadDLmodel()
    
#    face_cascade = cv2.CascadeClassifier(".\\haarcascades\\haarcascade_frontalface_default.xml")
#    eye_cascade = cv2.CascadeClassifier(".\\haarcascades\\haarcascade_eye.xml")
#
#    #Y=0
#    Y = np.zeros([50,1]) #50 faces
    
    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        frame = imutils.resize(frame, width=400)   
#        frame = live.test_steaming(frame=frame)
        #frame1 = live.liveFaceDetection_websteaming(frame=frame
#                                                    )
# 		# acquire the lock, set the output frame, and release the
# 		# lock
    with lock:
        outputFrame = frame.copy()
            
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')
            
@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")
      
      
# check to see if this is the main thread of execution
if __name__ == '__main__':
    
    t = threading.Thread(target=detect_motion, args=[32])
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host="192.168.1.54", port=8030, debug=True,
        threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()