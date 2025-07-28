from detector import EnhancedCrowdDetector

def main():
    detector = EnhancedCrowdDetector(
        model_path="models/yolov8n.pt",
        camera_index=0,
        save_analytics=True,
        use_better_face_detector=True  # Flag to use OpenCV-based face detector instead of FER
    )
    detector.run()

if __name__ == "__main__":
    main()
