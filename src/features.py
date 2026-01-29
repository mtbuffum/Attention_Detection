import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceTracker:
    def __init__(self):
        #---DETECTOR---
        #Initialize the media pipe face detector
        detector_model_path = 'blaze_face_short_range.tflite'
        detector_base_options = python.BaseOptions(
            model_asset_path= detector_model_path
        )

        detector_options = vision.FaceDetectorOptions(
            base_options=detector_base_options,
            running_mode=vision.RunningMode.IMAGE
        )
        #Our actual detector model using blaze face short downloaded from MP page
        self.detector= vision.FaceDetector.create_from_options(detector_options)

        #---LANDMARK---
        landmark_model_path= 'face_landmarker.task'
        landmark_base_options = python.BaseOptions(
            model_asset_path= landmark_model_path
        )

        landmark_options= vision.FaceLandmarkerOptions(
            base_options=landmark_base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,

        )

        self.landmarker = vision.FaceLandmarker.create_from_options(landmark_options)
            

    def detect_face(self,frame):
        """Converts the frame from BGR->RGB then 
        Return True if a face is detected in this frame"""

        #Convert frame from BGR(OPENCV) to RGB(MediaPipe)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Makes frame as a MP image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        #Resulting detection
        result = self.detector.detect(mp_image)

        return len(result.detections) > 0
    
    def get_landmarks(self,frame):
         #Convert frame from BGR(OPENCV) to RGB(MediaPipe)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        #process landmarks
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]
    
    def draw_landmarks(self, frame, landmarks):
        """Draw eye landmarks on the frame."""
        if landmarks is None:
            return frame

        h, w = frame.shape[:2]

        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        for idx, lm in enumerate(landmarks):
            x = int(lm.x * w)
            y = int(lm.y * h)

            if idx in LEFT_EYE or idx in RIGHT_EYE:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        return frame



    def close(self):
        #shuts down the fash_mesh
        self.detector.close()
        self.landmarker.close()

def euclidean_distance(p1,p2):
    """This Function is a helper function that will 
    compute the distance between Two inputted Landmarks"""
    #p1,p2 should be tuples (x,y)
    return math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)


def compute_ear(w,h, landmarks):
    """This is where we will compute the actual EAR value
    which is done by EAR = (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4))
    """
    #redefine EYE arrays (not necassarily most efficient way but no harm)
    LEFT_EYE = [(int(landmarks[33].x * w),int(landmarks[33].y* h)), 
                (int(landmarks[160].x* w),int(landmarks[160].y* h)), 
                (int(landmarks[158].x* w),int(landmarks[158].y* h)), 
                (int(landmarks[133].x* w),int(landmarks[133].y* h)), 
                (int(landmarks[153].x* w),int(landmarks[153].y* h)), 
                (int(landmarks[144].x* w),int(landmarks[144].y* h))]
    
    RIGHT_EYE = [(int(landmarks[362].x* w),int(landmarks[362].y* h)),
                 (int(landmarks[385].x* w),int(landmarks[385].y* h)),
                 (int(landmarks[387].x* w),int(landmarks[387].y* h)),
                 (int(landmarks[263].x* w),int(landmarks[263].y* h)),
                 (int(landmarks[373].x* w),int(landmarks[373].y* h)),
                 (int(landmarks[380].x* w),int(landmarks[380].y* h))]


    earL= ((euclidean_distance(LEFT_EYE[1],LEFT_EYE[5])+ euclidean_distance(LEFT_EYE[2],LEFT_EYE[4])) / (2 * euclidean_distance(LEFT_EYE[0],LEFT_EYE[3])))
    earR= ((euclidean_distance(RIGHT_EYE[1],RIGHT_EYE[5])+ euclidean_distance(RIGHT_EYE[2],RIGHT_EYE[4])) / (2 * euclidean_distance(RIGHT_EYE[0],RIGHT_EYE[3])))

    return earL, earR

def head_yaw_proxy(w,h, landmarks):
    """This function should be able to calculate 
    the proxy and yaw angles of the head"""

    #indices of key land marks
    LEFT_EYE = 33
    RIGHT_EYE = 263
    NOSE = 1

    #create them as pixel points
    leftx, lefty = int(landmarks[LEFT_EYE].x * w), int(landmarks[LEFT_EYE].y * h)
    rightx, righty = int(landmarks[RIGHT_EYE].x * w), int(landmarks[RIGHT_EYE].y * h)
    nosex, nosey = int(landmarks[NOSE].x * w), int(landmarks[NOSE].y * h)

    midx = (leftx + rightx) / 2
    eye_width = eye_width = ((rightx - leftx)**2 + (righty - lefty)**2) ** 0.5
   
    #to avoid small discrepancies 
    if eye_width < 1e-6:
        return 0.0
    
    yaw = (nosex-midx) / eye_width

    return yaw
       

        
        