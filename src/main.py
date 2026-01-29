from camera import Camera
from features import FaceTracker
from features import compute_ear
from features import head_yaw_proxy

from logger import setup_logger
from event_logger import EventLogger
import cv2
import time
import winsound

def main():
    #Create logging setup
    logger = setup_logger()
    event_log = EventLogger("attention_events.csv")
    prev_state = None
    #Create Objects
    cam = Camera()
    cam.open_camera()
    tracker = FaceTracker()
    
    #reset timing features
    eyes_closed_start = None

    #main loop reoccuring to show video feed
    while True:
        #get the frame from the camera ret is if it sucessfully gets it
        ret, frame = cam.get_frame()

        if not ret:
            print("Failed to read frame.")
            break

        #gets the height and width of the frame (used to calculate where pixels are in EAR)
        h,w = frame.shape[:2]
        #Uses face detector model to detect if face or not
        face_present= tracker.detect_face(frame)

        #display on screen if face present or not
        cv2.putText(
            frame,
            f"Face: {face_present}",
            (20,40),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            1,
            (0, 255, 0) if face_present else (0, 0, 255),
            2
        )

        #uses model to get landmarks found on face
        landmarks = tracker.get_landmarks(frame)

        #plot said landmarks
        frame = tracker.draw_landmarks(frame,landmarks)
        
        #fail safe in case no landmarks found
        if landmarks is not None:
            earL, earR = compute_ear(w,h,landmarks)
        
        #Calculate the EAR value
        if earL is not None and earR is not None:
            #average values
            ear_avg = (earR+earL) / 2
            #Display on screen in white showing specific value
            cv2.putText(
                frame,
                f"EAR: {ear_avg:.3f}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2
            )

            #Threshold of whether eyes are closed or not
            if ear_avg < .21:
                eyes_closed = True
            else:
                eyes_closed = False
            
            #eyes closed TIMING LOGIC
            now = time.time()
            if eyes_closed:
                if eyes_closed_start is None:
                    eyes_closed_start = now
                eyes_closed_duration = now - eyes_closed_start
            else:
                eyes_closed_start = None
                eyes_closed_duration = 0.0

            #Head PROXY logic
            if face_present:
             yaw = head_yaw_proxy(w, h, landmarks)
            if abs(yaw) > 0.25:
                head_turned = True
            else:
                head_turned = False

            #Head Turned Timing Logic
            if head_turned:
                if head_turned_start is None:
                    head_turned_start = now
                head_turned_duration = now - head_turned_start
            else:
                head_turned_start = None
                head_turned_duration = 0.0
            
            #Yaw Display
            cv2.putText(
                frame,
                f"YAW: {yaw:.3f}",
                (20,160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2
            )

            #Set Attention State
            if not face_present:
                attention_state = "AWAY"
            elif eyes_closed_duration > 2.0 or head_turned_duration > 3.0:
                attention_state = "DISTRACTED"
            else:
                attention_state = "FOCUSED"

            #Plays an alarm sound over and over until the eyes are open again (Small edition to prevent sleep)
            if eyes_closed_duration > 10.0 or head_turned_duration > 10.0:
                winsound.Beep(1000, 500)

            cv2.putText(
                frame,
                f"State: {attention_state}",
                (20,200),
                cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0) if not attention_state == "DISTRACTED" else (0, 0, 255),2
            )

            #Display whether eyes are closed or not
            cv2.putText(
            frame,
            f"Eyes Closed: {eyes_closed}",
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not eyes_closed else (0, 0, 255), 2
        )
            
        #  Event logging (only when state changes) 
        if attention_state != prev_state:
            event_log.log(
                state=attention_state,
                ear=ear_avg,
                eyes_sec=eyes_closed_duration,
                yaw=yaw,
                head_sec=head_turned_duration
            )
            logger.info(f"State change: {prev_state} -> {attention_state}")
            prev_state = attention_state
        #Display everything in window
        cv2.imshow("LandMarker", frame)

        #Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    

if __name__ == "__main__":
    main()