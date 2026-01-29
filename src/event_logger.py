import csv
import time

class EventLogger:
    def __init__(self, path="attention_events.csv"):
        self.path = path
        self._initialized = False
    
    def _init_file(self):
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "state", "ear", "eyes_closed_sec", "yaw", "head_turned_sec"])
        self._initialized = True
    
    def log(self, state, ear=None, eyes_sec=0.0, yaw=None, head_sec=0.0):
        if not self._initialized:
            self._init_file()
        
        ts = time.time()
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, state, ear, eyes_sec, yaw, head_sec])