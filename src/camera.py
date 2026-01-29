import cv2

class Camera:
    def __init__(self, index: int = 0):
        self.index = index
        self.cap = None
        

    def open_camera(self):
        """Opens the camera selected."""
        #This sets the connection to default video webcam of 0
        self.cap = cv2.VideoCapture(self.index)

        if not self.cap.isOpened():
            self.cap = None
            raise RuntimeError("Error: Could not open this video stream or file.")
            
        
        print("Webcam feed successfully opened.")

        
    def get_frame(self):
        """Gets a single frame so that we can process each frame at a time in main script"""
        if self.cap is None:
            raise RuntimeError("CAP is NONE")
        ret,frame = self.cap.read()
        if not ret:
            raise RuntimeError("Error: Falied to capture frame.")
                
        return ret,frame
    

    def release(self):
         """Releases the instance of our Video Capture"""
         self.cap.release()
         self.cap = None
         cv2.destroyAllWindows()



        
