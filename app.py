import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np


def load_model(model_path):
    return YOLO(model_path)


def draw_boxes(image, boxes, labels, confidences, class_colors):
    for box, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, box)
        label_text = f"{label} {conf:.2f}"
        color = class_colors.get(label, (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def process_video(video_path, person_model_path, ppe_model_path):
    person_model = load_model(person_model_path)
    ppe_model = load_model(ppe_model_path)

    class_colors = {
        "person": (0, 255, 0),
        "hard-hat": (255, 0, 0),
        "gloves": (0, 0, 255),
        "boots": (255, 0, 255),
        "vest": (0, 255, 255),
        "ppe-suit": (128, 0, 128),
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as output_file:
        output_path = output_file.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening the video file!")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    reduced_fps = original_fps // 2  # Reduce FPS to half for better detection

    frame_interval = int(original_fps / reduced_fps)

    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_path, fourcc, reduced_fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames based on reduced FPS
        if frame_count % frame_interval != 0:
            frame_count += 1
            continue

        original_frame = frame.copy()

        # Person detection
        person_results = person_model.predict(frame)
        person_bboxes = person_results[0].boxes.xyxy.cpu().numpy()
        person_confs = person_results[0].boxes.conf.cpu().numpy()
        person_labels = ["person"] * len(person_bboxes)

        if person_bboxes.size > 0:
            for idx, person_box in enumerate(person_bboxes):
                x1, y1, x2, y2 = map(int, person_box[:4])
                cropped_frame = original_frame[y1:y2, x1:x2]

                # PPE detection
                ppe_results = ppe_model.predict(cropped_frame)
                ppe_bboxes = ppe_results[0].boxes.xyxy.cpu().numpy()
                ppe_confs = ppe_results[0].boxes.conf.cpu().numpy()
                ppe_labels = [ppe_results[0].names[int(cls)] for cls in ppe_results[0].boxes.cls.cpu().numpy()]

                for ppe_box in ppe_bboxes:
                    ppe_box[0] += x1
                    ppe_box[1] += y1
                    ppe_box[2] += x1
                    ppe_box[3] += y1

                frame = draw_boxes(frame, [person_box], [person_labels[idx]], [person_confs[idx]], class_colors)
                frame = draw_boxes(frame, ppe_bboxes, ppe_labels, ppe_confs, class_colors)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return output_path


def process_webcam(person_model_path, ppe_model_path):
    person_model = load_model(person_model_path)
    ppe_model = load_model(ppe_model_path)

    class_colors = {
        "person": (0, 255, 0),
        "hard-hat": (255, 0, 0),
        "gloves": (0, 0, 255),
        "boots": (255, 0, 255),
        "vest": (0, 255, 255),
        "ppe-suit": (128, 0, 128),
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error accessing the webcam!")
        return

    st.info("Press 'Stop Webcam Detection' to quit the webcam feed.")

    frame_placeholder = st.empty()
    stop_button = st.button("Stop Webcam Detection")

    while not stop_button and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam.")
            break

        original_frame = frame.copy()
        resized_frame = cv2.resize(frame, (640, 480))

        # Person detection
        person_results = person_model.predict(resized_frame)
        person_bboxes = person_results[0].boxes.xyxy.cpu().numpy()
        person_confs = person_results[0].boxes.conf.cpu().numpy()
        person_labels = ["person"] * len(person_bboxes)

        if person_bboxes.size > 0:
            for idx, person_box in enumerate(person_bboxes):
                x1, y1, x2, y2 = map(int, person_box[:4])
                cropped_frame = original_frame[y1:y2, x1:x2]

                # PPE detection
                ppe_results = ppe_model.predict(cropped_frame)
                ppe_bboxes = ppe_results[0].boxes.xyxy.cpu().numpy()
                ppe_confs = ppe_results[0].boxes.conf.cpu().numpy()
                ppe_labels = [ppe_results[0].names[int(cls)] for cls in ppe_results[0].boxes.cls.cpu().numpy()]

                for ppe_box in ppe_bboxes:
                    ppe_box[0] += x1
                    ppe_box[1] += y1
                    ppe_box[2] += x1
                    ppe_box[3] += y1

                frame = draw_boxes(frame, [person_box], [person_labels[idx]], [person_confs[idx]], class_colors)
                frame = draw_boxes(frame, ppe_bboxes, ppe_labels, ppe_confs, class_colors)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB", use_container_width=True)

    cap.release()


def main():
    st.title("IndustryGuard™")
    st.subheader("Where AI Meets Workplace Safety! ")

    mode = st.sidebar.radio("Select Mode", ("Upload Video", "Webcam Detection"))

    person_model_path = "weights/person_det_model.pt"
    ppe_model_path = "weights/ppe_detection_model.pt"

    if mode == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(uploaded_file.read())
                temp_video_path = temp_video.name

            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    output_video = process_video(temp_video_path, person_model_path, ppe_model_path)
                    if output_video:
                        st.video(output_video)

    elif mode == "Webcam Detection":
        if st.button("Start Webcam Detection"):
            process_webcam(person_model_path, ppe_model_path)


if __name__ == "__main__":
    main()
