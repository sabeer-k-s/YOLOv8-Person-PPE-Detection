import os
import cv2
import argparse
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

def perform_inference(input_dir, output_dir, person_model_path, ppe_model_path):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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

    # Iterate through images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            original_image = image.copy()  # Copy original image for cropping

            # Person detection
            person_results = person_model.predict(image)
            person_bboxes = person_results[0].boxes.xyxy.cpu().numpy()  # Person bounding boxes
            person_confs = person_results[0].boxes.conf.cpu().numpy()  # Person confidence scores
            person_labels = ["person"] * len(person_bboxes)  # Person labels

            if person_bboxes.size > 0:
                for idx, person_box in enumerate(person_bboxes):
                    x1, y1, x2, y2 = map(int, person_box[:4])
                    cropped_image = original_image[y1:y2, x1:x2]  # Crop image to person bounding box

                    # PPE detection on the cropped image
                    ppe_results = ppe_model.predict(cropped_image)
                    ppe_bboxes = ppe_results[0].boxes.xyxy.cpu().numpy()  # PPE bounding boxes
                    ppe_confs = ppe_results[0].boxes.conf.cpu().numpy()  # PPE confidence scores
                    ppe_labels = [ppe_results[0].names[int(cls)] for cls in ppe_results[0].boxes.cls.cpu().numpy()]  # PPE labels

                    # Adjust PPE bounding boxes to the original image coordinates
                    for ppe_box in ppe_bboxes:
                        ppe_box[0] += x1
                        ppe_box[1] += y1
                        ppe_box[2] += x1
                        ppe_box[3] += y1

                    # Draw bounding boxes on the original image
                    image = draw_boxes(image, [person_box], [person_labels[idx]], [person_confs[idx]], class_colors)
                    image = draw_boxes(image, ppe_bboxes, ppe_labels, ppe_confs, class_colors)

            # Save the output image with drawn bounding boxes
            output_image_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_image_path, image)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Perform inference using person and PPE detection models.")
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("output_dir", type=str, help="Directory to save output images.")
    parser.add_argument("person_det_model", type=str, help="Path to the person detection model.")
    parser.add_argument("ppe_detection_model", type=str, help="Path to the PPE detection model.")
    
    args = parser.parse_args()
    perform_inference(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)

# To run this script, use the following command:
# python inference.py path/to/images path/to/output_dir weights/person_det_model.pt weights/ppe_detection_model.pt

