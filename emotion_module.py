# emotion_module.py
import cv2
import threading
import queue
import numpy as np
from deepface import DeepFace

class EmotionDetectorThread(threading.Thread):
    def __init__(self, results_dict, stop_event, use_better_face_detector=False):
        super().__init__()
        self.results_dict = results_dict
        self.stop_event = stop_event
        self.queue = queue.Queue(maxsize=10)
        self.use_better_face_detector = use_better_face_detector

        if use_better_face_detector:
            modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
            configFile = "models/deploy.prototxt"
            self.face_detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        else:
            self.face_detector = None

    def run(self):
        while not self.stop_event.is_set():
            try:
                frame, detections = self.queue.get(timeout=1)
                if frame is not None:
                    self._process_frame(frame, detections)
            except queue.Empty:
                continue

    def _process_frame(self, frame, detections):
        h, w = frame.shape[:2]
        for i in range(len(detections)):
            tracker_id = detections.tracker_id[i]
            if tracker_id is None:
                continue
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            face_crop = frame[y1:y2, x1:x2]

            try:
                analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                if isinstance(analysis, list):
                    emotion = analysis[0]['dominant_emotion']
                else:
                    emotion = analysis['dominant_emotion']
                self.results_dict[tracker_id] = emotion
            except Exception:
                continue

    def stop(self):
        self.stop_event.set()
        self.join()