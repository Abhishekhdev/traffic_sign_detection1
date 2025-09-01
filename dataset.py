import cv2
import os
from glob import glob
import argparse

def create_dataset(output_dir, classes, img_size=(100, 100), num_images=100, video_path=None, image_path=None):
    """
    Collect a dataset of traffic sign images from live webcam, video, or existing image folder.
    
    Parameters:
    - output_dir: Directory to save the dataset.
    - classes: List of traffic sign classes.
    - img_size: Size of each image in the dataset.
    - num_images: Number of images to capture per class (for webcam/video).
    - video_path: Path to a video file to extract frames (optional).
    - image_path: Path to a folder containing images (optional).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label in classes:
        class_dir = os.path.join(output_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Collecting images for class: {label}")

        # Collect images from webcam or video
        if video_path or not image_path:
            cap = cv2.VideoCapture(0 if video_path is None else video_path)
            count = 0
            while count < num_images:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from video/camera.")
                    break

                frame_resized = cv2.resize(frame, img_size)
                cv2.imshow(f"Collecting {label} ({count}/{num_images})", frame)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('c'):  # Press 'c' to capture the image
                    cv2.imwrite(os.path.join(class_dir, f"{count}.jpg"), frame_resized)
                    count += 1
                elif key & 0xFF == ord('q'):  # Press 'q' to quit early
                    break

            cap.release()
            cv2.destroyAllWindows()

        # Collect images from an existing folder
        if image_path:
            image_files = glob(os.path.join(image_path, "*"))
            count = 0
            for img_file in image_files:
                if count >= num_images:
                    break

                img = cv2.imread(img_file)
                if img is None:
                    continue

                frame_resized = cv2.resize(img, img_size)
                cv2.imwrite(os.path.join(class_dir, f"{count}.jpg"), frame_resized)
                count += 1
            print(f"Collected {count} images for class: {label} from folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Creator for Traffic Sign Detection")
    parser.add_argument("--output_dir", type=str, default="D:/traffic_sign_detection/traffic_signs_dataset", help="Output directory for the dataset")
    parser.add_argument("--source", type=str, choices=['webcam', 'video', 'images'], default='webcam', help="Source of images: webcam, video, or images folder")
    parser.add_argument("--classes", nargs="+", required=True, help="List of traffic sign classes")
    parser.add_argument("--img_size", type=int, nargs=2, default=(100, 100), help="Size of the output images")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to collect per class")
    parser.add_argument("--video_path", type=str, help="Path to a video file (optional)")
    parser.add_argument("--image_path", type=str, help="Path to an image folder (optional)")

    args = parser.parse_args()

    # Call the dataset creation function based on the source
    if args.source == 'webcam':
        create_dataset(args.output_dir, args.classes, img_size=args.img_size, num_images=args.num_images)
    elif args.source == 'video':
        create_dataset(args.output_dir, args.classes, img_size=args.img_size, num_images=args.num_images, video_path=args.video_path)
    elif args.source == 'images':
        create_dataset(args.output_dir, args.classes, img_size=args.img_size, image_path=args.image_path)
