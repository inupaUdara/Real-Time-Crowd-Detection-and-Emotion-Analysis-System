import time
import cv2
import numpy as np
import threading
import json
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
import supervision as sv
from emotion_module import EmotionDetectorThread
from analytics import AnalyticsManager
from visualizer import Visualizer
import os

os.makedirs("analytics", exist_ok=True)

class EnhancedCrowdDetector:
    def __init__(self, model_path, camera_index=0, save_analytics=True, use_better_face_detector=False):
        self.model = YOLO(model_path)
        self.device = self._get_device()
        print(f"Using device: {self.device}")

        self.camera = cv2.VideoCapture(camera_index)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        self.fps = self.camera.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=self.fps
        )

        self.line_counter = sv.LineZone(
            start=sv.Point(int(self.width * 0.1), int(self.height * 0.4)),
            end=sv.Point(int(self.width * 0.9), int(self.height * 0.4))
        )

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.4)
        self.line_annotator = sv.LineZoneAnnotator(thickness=3)

        self.save_analytics = save_analytics
        self.person_tracks = defaultdict(lambda: {
            'first_seen': None,
            'last_seen': None,
            'emotions': deque(maxlen=30),
            'positions': deque(maxlen=30)
        })

        self.frame_times = deque(maxlen=30)
        self.detection_counts = deque(maxlen=100)
        self.total_emotion_counts = defaultdict(int)  # Track total emotion counts

        self.density_zones = self._setup_density_zones()

        self.emotion_results = {}
        self.emotion_stop_event = threading.Event()
        self.emotion_detector = EmotionDetectorThread(
            results_dict=self.emotion_results,
            stop_event=self.emotion_stop_event,
            use_better_face_detector=use_better_face_detector
        )

    def _get_device(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.model.to("cuda")
                return "cuda"
            elif torch.backends.mps.is_available():
                self.model.to("mps")
                return "mps"
        except:
            pass
        return "cpu"

    def _setup_density_zones(self):
        return {
            1: sv.PolygonZone(
                polygon=np.array([
                    [0, 0],
                    [self.width // 2, 0],
                    [self.width // 2, self.height // 2],
                    [0, self.height // 2]
                ])
            ),
            2: sv.PolygonZone(
                polygon=np.array([
                    [self.width // 2, 0],
                    [self.width, 0],
                    [self.width, self.height // 2],
                    [self.width // 2, self.height // 2]
                ])
            )
        }

    def process_frame(self, frame):
        results = self.model(frame, conf=0.5, classes=0, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]
        detections = self.tracker.update_with_detections(detections)

        self.line_counter.trigger(detections)
        current_time = datetime.now()
        active_ids = set()

        if not self.emotion_detector.queue.full():
            self.emotion_detector.queue.put((frame.copy(), detections))

        for i in range(len(detections)):
            tracker_id = detections.tracker_id[i]
            if tracker_id is not None:
                active_ids.add(tracker_id)
                track = self.person_tracks[tracker_id]
                if track['first_seen'] is None:
                    track['first_seen'] = current_time
                track['last_seen'] = current_time

                bbox = detections.xyxy[i]
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                track['positions'].append(center)

                if tracker_id in self.emotion_results:
                    emotion = self.emotion_results[tracker_id]
                    track['emotions'].append(emotion)
                    self.total_emotion_counts[emotion] += 1

        self.detection_counts.append(len(detections))
        self.frame_times.append(current_time)

        analytics = AnalyticsManager(
            save_dir="analytics",
            save_analytics=self.save_analytics
        )

        for i in range(len(detections)):
            tracker_id = detections.tracker_id[i]
            if tracker_id is not None:
                analytics.update_tracking_data(tracker_id, current_time)
                for zone_id, zone in self.density_zones.items():
                    trigger_result = zone.trigger(detections=detections)
                    if i < len(trigger_result) and trigger_result[i]:
                        analytics.record_zone_entry(tracker_id, zone_id, current_time)

        current_emotions = {tid: self.emotion_results.get(tid) for tid in active_ids if tid in self.emotion_results}
        emotion_counts = {}
        for emo in current_emotions.values():
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

        annotated = Visualizer().draw_detections(
            frame=frame.copy(),
            detections=[{"bbox": detections.xyxy[i], "id": detections.tracker_id[i]} for i in range(len(detections)) if detections.tracker_id[i] is not None],
            emotions=self.emotion_results,
            zone_density={zone_id: zone.current_count for zone_id, zone in self.density_zones.items()},
            emotion_counts=emotion_counts
        )

        return annotated, analytics

    def save_analytics_to_file(self, analytics):
        if not self.save_analytics:
            return
        os.makedirs("analytics", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analytics/crowd_analytics_{timestamp}.json"
        analytics.save()

    def run(self):
        print("Starting detection... Press 'q' to quit.")
        self.emotion_detector.start()

        last_save_time = time.time()
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Camera read error")
                    break

                annotated_frame, analytics = self.process_frame(frame)

                display_frame = annotated_frame
                if self.width > 1280 or self.height > 720:
                    scale = min(1280 / self.width, 720 / self.height)
                    display_frame = cv2.resize(annotated_frame, (int(self.width * scale), int(self.height * scale)))

                cv2.imshow("Crowd Detection", display_frame)

                if time.time() - last_save_time > 60:
                    self.save_analytics_to_file(analytics)
                    last_save_time = time.time()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_analytics_to_file(analytics)
                elif key == ord('r'):
                    self.line_counter.in_count = 0
                    self.line_counter.out_count = 0

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        print("Cleaning up...")
        self.camera.release()
        cv2.destroyAllWindows()
        self.emotion_stop_event.set()
        self.emotion_detector.join()
        if self.save_analytics:
            summary = {
                'session_end': datetime.now().isoformat(),
                'total_unique_persons': len(self.person_tracks),
                'total_in': self.line_counter.in_count,
                'total_out': self.line_counter.out_count,
                'avg_detection_count': float(np.mean(self.detection_counts)) if self.detection_counts else 0,
                'total_emotion_counts': dict(self.total_emotion_counts)
            }
            with open('analytics/session_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)