import cv2

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_detections(self, frame, detections, emotions, zone_density=None, emotion_counts=None):
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            object_id = det["id"]
            label = f"ID: {object_id}"
            if object_id in emotions:
                label += f" | {emotions[object_id]}"

            color = (0, 255, 0) if emotions.get(object_id) != "angry" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), self.font, 0.5, color, 2)

        if zone_density:
            y_pos = 30
            for zone_id, count in zone_density.items():
                cv2.putText(frame, f"Zone {zone_id}: {count} people", (10, y_pos), self.font, 0.6, (255, 255, 0), 2)
                y_pos += 20

        if emotion_counts:
            y_pos = y_pos + 20 if zone_density else 30
            for emotion, count in emotion_counts.items():
                cv2.putText(frame, f"{emotion}: {count}", (10, y_pos), self.font, 0.6, (255, 255, 255), 2)
                y_pos += 20

        return frame