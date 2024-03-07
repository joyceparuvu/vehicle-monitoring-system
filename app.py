


from flask import Flask, render_template, request
import cv2
import dlib
import time
import threading
import math
import os
import time
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



carCascade = cv2.CascadeClassifier('myhaar.xml')
WIDTH = 1280
HEIGHT = 720

# Rest of your code...


def detectMultipleObjects(video_path):


### input and output path
  INPUT_VIDEO_PATH = video_path
  OUTPUT_VIDEO_PATH = "output.mp4"

### model
  model = YOLO("yolov8s")
  video = cv2.VideoCapture(INPUT_VIDEO_PATH)

  VIDEO_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  VIDEO_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  VIDEO_FPS = video.get(cv2.CAP_PROP_FPS)
  output_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (640, 480))

### first frame
  _, first_frame = video.read()
  first_frame = cv2.resize(first_frame, (640,480))
  prevgray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

### create mask for optical flow
  mask = np.zeros_like(first_frame)
  mask[..., 1] = 255 #set image saturation to maximum

### frame_count
  frame_count = 0

  while True:
      ret, frame = video.read()
      if not ret:
          break
      frame_count += 1
      frame = cv2.resize(frame, (640, 480))
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      frame2 = frame.copy()

    ### start
      start = time.time()

    ### detect objects in a frame
      res = model.predict(frame, verbose=False, imgsz=320, conf=0.75)
      detections = res[0].boxes.data
      class_labels = res[0].names

      if frame_count%10 == 1:
        ### find optic flow (angle, motion)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 6, 10, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        magnitude, angle = cv2.cartToPolar(flow[...,0], flow[..., 1])

        ### set prev frame
        prevgray = gray

      for detection in detections:
        if detection[-2] >= 0.5:
            x1, y1, x2, y2 = detection[:-2]

            width = int(x2-x1)
            height = int(y2-y1)
            area = 0
            ct = 0

            for i in range(int(x1)+5, int(x2)-5, 10):
                for j in range(int(y1)+5, int(y2)-5, 10):
                    mag, ang = flow[j, i]
                    area += 1
                    if abs(mag) >= 2:
                        ct += 1

            if (abs(ct) >= 40) or (abs(ct) >= area*0.5):
                cv2.rectangle(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame2, class_labels[int(detection[-1])], (int(x1+1), int(y1-5)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

    ### end
      end = time.time()
      frame_rate = 1/(end - start)

      print(f"fps: {frame_rate:.2f}")
      cv2.imshow("Original Video", frame)
      cv2.imshow("Moving Object Detection Video", frame2)

    ### write to output video
      output_video.write(frame2)
      if cv2.waitKey(10) == ord('q'):
        break

  print("Output video generated")
  video.release()
  cv2.destroyAllWindows()

def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 18
    speed = d_meters * fps * 3.6
    return speed

def trackMultipleObjects(video_path):
    video = cv2.VideoCapture(video_path)
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0
    
    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000
    
    while True:
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break
        
        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()
        
        frameCounter = frameCounter + 1
        
        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)
            
            if trackingQuality < 7:
                carIDtoDelete.append(carID)
                
        for carID in carIDtoDelete:
            print ('Removing carID ' + str(carID) + ' from list of trackers.')
            print ('Removing carID ' + str(carID) + ' previous location.')
            print ('Removing carID ' + str(carID) + ' current location.')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)
        
        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
            
            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)
            
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h
                
                matchCarID = None
            
                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()
                    
                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())
                    
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h
                
                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID
                
                if matchCarID is None:
                    print ('Creating new tracker ' + str(currentCarID))
                    
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1
        
        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()
                    
            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())
            
            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
            
            carLocation2[carID] = [t_x, t_y, t_w, t_h]
        
        end_time = time.time()
        
        if not (end_time == start_time):
            fps = 1.0/(end_time - start_time)
        
        for i in carLocation1.keys():  
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]
        
                carLocation1[i] = [x2, y2, w2, h2]
        
                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

                    if speed[i] != None and y1 >= 180:
                        cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    
        cv2.imshow('result', resultImage)

        if cv2.waitKey(33) == 27:
            break
    
    cv2.destroyAllWindows()

def detectDeblurring(image_path):
  img = cv2.imread(image_path)
  print("Reading image from:", image_path)
  averaging = cv2.blur(img, (21, 21))
  gaussian = cv2.GaussianBlur(img, (21, 21), 0)
  median = cv2.medianBlur(img, 5)
  bilateral = cv2.bilateralFilter(img, 9, 350, 350)

  cv2.imshow("Original image", img)
  #cv2.imshow("Averaging", averaging)
  #cv2.imshow("Gaussian", gaussian)
  cv2.imshow("Median", median)
  cv2.imshow("Bilateral", bilateral)

  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  
  
def detectNumberPlate():
    harcascade = "model/haarcascade_russian_plate_number.xml"

    cap = cv2.VideoCapture(0)

    cap.set(3, 640) # width
    cap.set(4, 480) #height

    min_area = 500
    count = 0

    while True:
      success, img = cap.read()

      plate_cascade = cv2.CascadeClassifier(harcascade)
      img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

      for (x,y,w,h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y+h, x:x+w]
            cv2.imshow("ROI", img_roi)


    
      cv2.imshow("Result", img)

      if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)
        cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results",img)
        cv2.waitKey(500)
        count += 1
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
     
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/execute_script', methods=['POST'])
def execute_script():
    if 'video' not in request.files:
        return 'No video uploaded', 400
    video_file = request.files['video']
    if video_file.filename == '':
        return 'No selected video file', 400
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
    video_file.save(video_path)
    trackMultipleObjects(video_path)
    return render_template('index.html')


@app.route('/execute_script2', methods=['POST'])
def execute_script2():
    if 'video' not in request.files:
        return 'No video uploaded', 400
    video_file = request.files['video']
    if video_file.filename == '':
        return 'No selected video file', 400
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
    video_file.save(video_path)
    detectMultipleObjects(video_path)
    return render_template('index.html')

@app.route('/execute_script3', methods=['POST'])


def execute_script3():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    image_file = request.files['image']
    if image_file.filename == '':
        return 'No selected image file', 400
   

# Accept either PNG or JPG images
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    filename, file_extension = os.path.splitext(image_file.filename)
    if file_extension[1:].lower() not in allowed_extensions:
      return 'Invalid file extension. Only PNG and JPG files are allowed.', 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_image' + file_extension)

    #image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_image.jpg')
    image_file.save(image_path)
    detectDeblurring(image_path)
    return render_template('index.html')


@app.route('/execute_script4', methods=['POST'])
def execute_script4():
    
    detectNumberPlate()
    return render_template('index.html')

@app.route('/aboutus') 
def aboutus():
    return render_template('aboutus.html') 

@app.route('/help') 
def help(): 
    return render_template('help.html')

@app.route('/index1') 
def index1():
    return render_template('index.html') 


if __name__ == '__main__':
    app.run(debug=True)
