import os
import xml.etree.ElementTree as ET
import argparse

# Predefined class labels for the conversion
# These labels must match exactly with those used in the Pascal VOC XML files
# Remove person if converting only PPE objects
class_labels = ['person','hard-hat', 'gloves', 'boots', 'vest', 'ppe-suit']

# Function to convert Pascal VOC format to YOLO format
def convert_voc_to_yolo(voc_xml_file, img_width, img_height):
    """
    Convert a single Pascal VOC XML file to YOLO format.
    
    Parameters:
    - voc_xml_file: Path to the Pascal VOC XML file.
    - img_width: Width of the image (used for normalization).
    - img_height: Height of the image (used for normalization).

    Returns:
    - yolo_annotations: List of YOLO formatted annotations for the image.
    """
    tree = ET.parse(voc_xml_file)
    root = tree.getroot()
    
    yolo_annotations = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        if label not in class_labels:
            continue  # Skip labels not in predefined class labels
        
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # Calculate YOLO format coordinates (normalized)
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # Determine the class ID based on the label
        class_id = class_labels.index(label)
        
        # Append the YOLO formatted annotation to the list
        yolo_annotations.append((class_id, x_center, y_center, width, height))
    
    return yolo_annotations

# Function to save YOLO annotations to a text file
def save_yolo_annotations(yolo_annotations, yolo_txt_file):
    """
    Save YOLO formatted annotations to a text file.
    
    Parameters:
    - yolo_annotations: List of YOLO formatted annotations.
    - yolo_txt_file: Path to the output YOLO .txt file.
    """
    with open(yolo_txt_file, 'w') as file:
        for annotation in yolo_annotations:
            class_id, x_center, y_center, width, height = annotation
            # Write each annotation in the format: class_id x_center y_center width height
            file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Function to process all XML files in the directory and convert them to YOLO format
def process_directory(input_dir, output_dir):
    """
    Process all Pascal VOC XML files in the input directory and convert them to YOLO format.
    
    Parameters:
    - input_dir: Directory containing Pascal VOC XML files.
    - output_dir: Directory where YOLO annotations will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it does not exist

    # Assuming all images are of size 640x640
    img_width = 640
    img_height = 640

    for xml_file in os.listdir(input_dir):
        if not xml_file.endswith('.xml'):
            continue  # Skip files that are not XML
        
        voc_xml_file = os.path.join(input_dir, xml_file)
        
        # Convert VOC annotation to YOLO format
        yolo_annotations = convert_voc_to_yolo(voc_xml_file, img_width, img_height)
        
        # Define the output path for the YOLO formatted annotation file
        yolo_txt_file = os.path.join(output_dir, xml_file.replace('.xml', '.txt'))
        
        # Save YOLO annotations to the text file
        save_yolo_annotations(yolo_annotations, yolo_txt_file)
        print(f"Converted {xml_file} to {yolo_txt_file}")

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pascal VOC annotations to YOLO format.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing Pascal VOC XML annotations.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where YOLO annotations will be saved.")

    args = parser.parse_args()
    
    # Process the directory based on user input
    process_directory(args.input_dir, args.output_dir)

# Example command to run the script:
# python pascalVOC_to_yolo.py path/to/input/xml/files path/to/output/yolo/files
