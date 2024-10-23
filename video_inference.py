import os
import cv2
from ultralytics import YOLO

def load_model(model_path):
    # Load the YOLO model from the given model path
    return YOLO(model_path)

def draw_boxes(image, boxes, labels, confidences, class_colors):
    # Draw bounding boxes and labels on the image
    for box, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
        label_text = f"{label} {conf:.2f}"  # Create label text with confidence score
        color = class_colors.get(label, (0, 255, 0))  # Get color for the label, default to green
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # Draw rectangle on the image
        cv2.putText(image, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Put label text on the image
    return image

def perform_inference_on_video(video_path, output_video_path, person_model_path, ppe_model_path):
    # Load person and PPE detection models
    person_model = load_model(person_model_path)
    ppe_model = load_model(ppe_model_path)

    # Define colors for each class
    class_colors = {
        "person": (0, 255, 0),       # Green
        "hard-hat": (255, 0, 0),     # Blue
        "gloves": (0, 0, 255),       # Red
        "boots": (255, 0, 255),      # Magenta
        "vest": (0, 255, 255),       # Yellow
        "ppe-suit": (128, 0, 128),   # Purple
    }

    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object to save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()  # Copy original frame for cropping

        # Person detection
        person_results = person_model.predict(frame)
        person_bboxes = person_results[0].boxes.xyxy.cpu().numpy()  # Person bounding boxes
        person_confs = person_results[0].boxes.conf.cpu().numpy()  # Person confidence scores
        person_labels = ["person"] * len(person_bboxes)  # Person labels

        if person_bboxes.size > 0:
            for idx, person_box in enumerate(person_bboxes):
                x1, y1, x2, y2 = map(int, person_box[:4])
                cropped_frame = original_frame[y1:y2, x1:x2]  # Crop image to person bounding box

                # PPE detection on the cropped image
                ppe_results = ppe_model.predict(cropped_frame)
                ppe_bboxes = ppe_results[0].boxes.xyxy.cpu().numpy()  # PPE bounding boxes
                ppe_confs = ppe_results[0].boxes.conf.cpu().numpy()  # PPE confidence scores
                ppe_labels = [ppe_results[0].names[int(cls)] for cls in ppe_results[0].boxes.cls.cpu().numpy()]  # PPE labels

                # Adjust PPE bounding boxes to the original image coordinates
                for ppe_box in ppe_bboxes:
                    ppe_box[0] += x1
                    ppe_box[1] += y1
                    ppe_box[2] += x1
                    ppe_box[3] += y1

                # Draw bounding boxes on the original frame
                frame = draw_boxes(frame, [person_box], [person_labels[idx]], [person_confs[idx]], class_colors)
                frame = draw_boxes(frame, ppe_bboxes, ppe_labels, ppe_confs, class_colors)

        # Write the processed frame to the output video
        out.write(frame)

    # Release video objects
    cap.release()
    out.release()
    print(f"Video saved at {output_video_path}")

# Example of usage:
video_path = "input_video.mp4"  # Input video file
output_video_path = "output_video.mp4"  # Output video file
person_model_path = "weights/person_det_model.pt"  # Path to the person detection model
ppe_model_path = "weights/ppe_detection_model.pt"  # Path to the PPE detection model

perform_inference_on_video(video_path, output_video_path, person_model_path, ppe_model_path)
