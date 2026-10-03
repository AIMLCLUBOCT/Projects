"""
Real-Time Safety Equipment & Helmet Detection
AIML Club — Oriental College of Technology, Bhopal

An intermediate-level Computer Vision project teaching:
1. Object detection principles using YOLO (You Only Look Once).
2. Bounding box coordinates, Confidence scores, and Non-Maximum Suppression (NMS).
3. Intersection-over-Union (IoU) calculation for spatial overlap verification.
4. Edge AI deployment considerations (FPS, latency, webcam feeds).

Supports:
- Live YOLO detection with OpenCV & Ultralytics when installed.
- Pure Python simulation pipeline explaining IoU and detection parsing with zero dependencies.
"""

import sys
import time

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import cv2
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def calculate_iou(boxA, boxB):
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes.
    Format: [x1, y1, x2, y2]
    """
    # Coordinates of intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    # Compute areas of both individual boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = float(boxA_area + boxB_area - inter_area)
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def run_simulation():
    """
    Educational simulation demonstrating how detection models process
    camera frames, parse bounding boxes, and evaluate safety compliance.
    """
    print("\n💡 Running Educational Simulation Pipeline (Zero External Dependencies):")
    print("   • Framework: YOLO Architecture Intuition")
    print("   • Scenario:  College Campus Two-Wheeler Safety Checkpoint")

    # Simulated detection frames
    simulated_frames = [
        {
            "frame_id": 1,
            "detections": [
                {"class_name": "person", "box": [120, 80, 280, 440], "conf": 0.94},
                {"class_name": "helmet", "box": [140, 85, 230, 170], "conf": 0.91},
                {"class_name": "motorcycle", "box": [80, 260, 360, 480], "conf": 0.89},
            ]
        },
        {
            "frame_id": 2,
            "detections": [
                {"class_name": "person", "box": [150, 90, 310, 450], "conf": 0.96},
                {"class_name": "motorcycle", "box": [100, 270, 380, 490], "conf": 0.92},
                # Helmet missing!
            ]
        },
        {
            "frame_id": 3,
            "detections": [
                {"class_name": "person", "box": [110, 75, 270, 430], "conf": 0.93},
                {"class_name": "helmet", "box": [130, 80, 220, 165], "conf": 0.88},
                {"class_name": "vest", "box": [125, 175, 260, 320], "conf": 0.85},
            ]
        }
    ]

    print("\n🎬 Processing Campus Camera Stream (Simulated Frames):")
    print("-" * 65)

    for frame in simulated_frames:
        f_id = frame["frame_id"]
        dets = frame["detections"]
        has_person = any(d["class_name"] == "person" for d in dets)
        has_helmet = any(d["class_name"] == "helmet" for d in dets)
        has_bike = any(d["class_name"] == "motorcycle" for d in dets)

        # Compliance rule: If riding motorcycle, helmet must be worn
        if has_bike and has_person:
            if has_helmet:
                status = "✅ COMPLIANT (Helmet Worn on Two-Wheeler)"
            else:
                status = "🚨 VIOLATION DETECTED (Rider Without Helmet)"
        elif has_helmet:
            status = "✅ COMPLIANT (Safety Gear Equipped)"
        else:
            status = "ℹ️  PEDESTRIAN (No Two-Wheeler Detected)"

        print(f"\n[Frame #{f_id}] Objects Detected: {len(dets)}")
        for d in dets:
            print(f"   • {d['class_name'].capitalize():<12} | Conf: {d['conf'] * 100:.1f}% | Box: {d['box']}")
        print(f"Compliance Assessment: {status}")

    # IoU demonstration
    box1 = [120, 80, 250, 400]
    box2 = [130, 90, 260, 410]
    box3 = [300, 300, 400, 400]

    print("\n" + "=" * 65)
    print("📐 Spatial Overlap (IoU) Demonstration:")
    print("=" * 65)
    iou_high = calculate_iou(box1, box2)
    iou_zero = calculate_iou(box1, box3)
    print(f"   • Overlapping Bounding Boxes IoU: {iou_high:.4f} (High overlap -> candidate for NMS)")
    print(f"   • Non-Overlapping Bounding Boxes IoU: {iou_zero:.4f} (Distinct detected entities)")


def run_yolo_detection(source=0):
    """
    Runs actual Ultralytics YOLO inference on live webcam or video file.
    """
    print("\n🎥 Initializing Ultralytics YOLOv8 Inference...")
    model = YOLO("yolov8n.pt")  # Nano weights for real-time inference
    print("   • Weights loaded: YOLOv8 Nano")
    print(f"   • Connecting to video source: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"   ❌ Error: Unable to access camera source {source}.")
        print("   Falling back to simulation.")
        run_simulation()
        return

    print("   Press 'q' in the OpenCV preview window to exit.")
    fps_start = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, verbose=False)[0]

        # Draw bounding boxes and labels
        annotated_frame = results.plot()

        # Calculate FPS
        elapsed = time.time() - fps_start
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("AIML Club OCT - Safety Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("=" * 65)
    print("⛑️  AIML Club OCT — Safety Equipment & Helmet Detection")
    print("=" * 65)

    if YOLO_AVAILABLE:
        run_yolo_detection(source=0)
    else:
        print("\nℹ️  Notice: `ultralytics` and `opencv-python` are not installed.")
        print("   To enable live webcam/video inference: pip install -r requirements.txt")
        run_simulation()

    print("\n" + "=" * 65)
    print("🎉 Project pipeline executed! Ready for fine-tuning & contributions.")
    print("=" * 65)


if __name__ == "__main__":
    main()
