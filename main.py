import cv2
from time import time
import numpy as np
# Removed: from ultralytics.solutions import BaseSolution
from ultralytics import YOLO # Import YOLO
from ultralytics.utils.plotting import Annotator, colors
from datetime import datetime
from paddleocr import PaddleOCR
import csv
import os
import math # Added for distance calculation

# --- Configuration ---
# IMPORTANT: Calibrate this value for your specific camera setup and road perspective!
# It represents how many pixels correspond to one meter in the video frame
# at the typical distance where vehicles are tracked for speed.
# This is crucial for accurate speed calculation.
# You might need different factors for horizontal vs. vertical movement
# depending on the camera angle. For simplicity, we use one factor here.
# Example: If a car known to be 4.5 meters long appears as 150 pixels long in the video.
# pixels_per_meter = 150 / 4.5 = 33.3
PIXELS_PER_METER = 20.0 # <---- !! ADJUST THIS VALUE !!

CSV_FILENAME = "numberplates_speedd.csv"
MODEL_PATH = r"best.pt" # Path to your YOLO model
VIDEO_PATH = r'inputt.mp4' # Path to your video file
CONFIDENCE_THRESHOLD = 0.4 # Minimum confidence to consider a detection
PROCESS_EVERY_N_FRAMES = 3 # Process every Nth frame to speed up (adjust as needed)
FRAME_WIDTH = 1020
FRAME_HEIGHT = 500

# --- Class Definition ---

class SpeedEstimator:
    def __init__(self, model_path, class_names, csv_filename=CSV_FILENAME):
        # Load the YOLO model
        self.model = YOLO(model_path)
        # Store class names provided by the model or manually
        self.class_names = class_names if class_names else self.model.names

        # Tracking and Speed State
        self.track_history = {} # Stores {track_id: [(timestamp, center_x, center_y), ...]}
        self.speed_estimates = {} # Stores {track_id: speed_kph}
        self.logged_ids = set()   # Stores track_ids that have been saved to CSV in the current run

        # OCR Initialization
        # Added show_log=False to reduce PaddleOCR console spam
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

        # CSV Initialization
        self.csv_filename = csv_filename
        self.csv_header = ["date", "time", "track_id", "class_name", "speed_kmh", "numberplate"]
        self._initialize_csv()

        print("SpeedEstimator initialized.")
        print(f"Using class names: {self.class_names}")
        if not self.class_names:
             print("Warning: Class names not found or provided. Object class logging might be incorrect.")


    def _initialize_csv(self):
        """Initializes the CSV file and writer."""
        file_exists = os.path.exists(self.csv_filename)
        # Check if file exists and is not empty
        is_empty = os.path.getsize(self.csv_filename) == 0 if file_exists else True

        try:
            # Open in append mode ('a'), create if doesn't exist
            # Use newline='' to prevent extra blank rows in CSV
            # Use utf-8 encoding for broader character support
            self.csv_file = open(self.csv_filename, 'a', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)

            # Write header only if the file is newly created or was empty
            if not file_exists or is_empty:
                self.csv_writer.writerow(self.csv_header)
                self.csv_file.flush() # Ensure header is written immediately
                print(f"CSV file '{self.csv_filename}' created/initialized with header.")
            else:
                print(f"Appending to existing CSV file '{self.csv_filename}'.")

        except IOError as e:
            print(f"FATAL ERROR: Could not open or write to CSV file '{self.csv_filename}': {e}")
            raise # Re-raise the exception to stop execution if CSV can't be opened

    def perform_ocr(self, image_array):
        """Performs OCR on the given image array and returns cleaned text."""
        if image_array is None or image_array.size == 0:
            return "" # Return empty string for empty images

        # Ensure image is a BGR numpy array
        if not isinstance(image_array, np.ndarray):
            # print(f"Warning: OCR input type is {type(image_array)}, expected numpy array.")
            return ""
        if image_array.ndim == 2: # Grayscale
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        elif image_array.ndim != 3 or image_array.shape[2] != 3:
            # print(f"Warning: OCR input image shape {image_array.shape} not standard BGR.")
            return "" # Cannot process non-3-channel images reliably

        try:
            # Perform OCR
            results = self.ocr.ocr(image_array, cls=True) # cls=True helps orientation
            if results and results[0]: # Check if results are valid and contain data
                # Extract text pieces
                extracted_texts = [line[1][0] for line in results[0] if line and len(line) > 1 and len(line[1]) > 0]
                full_text = ' '.join(extracted_texts).strip()
                # Basic cleaning: Keep alphanumeric, convert to uppercase
                cleaned_text = ''.join(filter(str.isalnum, full_text)).upper()
                return cleaned_text
            else:
                return "" # No text found
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            # Optional: Save problematic image for debugging
            # cv2.imwrite(f"debug_ocr_error_{int(time())}.png", image_array)
            return "" # Return empty string on error


    def save_to_csv(self, date_str, time_str, track_id, class_name, speed, numberplate):
        """Saves a data row to the CSV file."""
        try:
            # Ensure speed is formatted correctly
            speed_value = float(speed) if speed is not None else 0.0
            # Prepare row data
            row = [date_str, time_str, track_id, class_name, f"{speed_value:.2f}", numberplate]
            # Write to CSV and flush
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        except IOError as e:
            print(f"Error writing to CSV file '{self.csv_filename}': {e}")
            # Decide if you want to stop execution or just log the error
            # raise e # Uncomment to stop if CSV writing fails critically
        except Exception as e:
            print(f"An unexpected error occurred during CSV saving: {e}")

    def _estimate_speed_for_track(self, track_id, current_time):
        """Calculates speed for a given track_id based on its history."""
        speed_kph = None
        if track_id in self.track_history and len(self.track_history[track_id]) >= 2:
            # Get the last two recorded points for this track
            last_time, last_x, last_y = self.track_history[track_id][-1]
            prev_time, prev_x, prev_y = self.track_history[track_id][-2]

            time_diff = last_time - prev_time
            # Avoid division by zero or extremely small time intervals
            if time_diff > 0.01:
                # Calculate distance in pixels
                pixel_dist = math.sqrt((last_x - prev_x)**2 + (last_y - prev_y)**2)

                # Convert distance to meters using the calibration factor
                meter_dist = pixel_dist / PIXELS_PER_METER

                # Calculate speed in meters per second
                speed_mps = meter_dist / time_diff

                # Convert speed to kilometers per hour
                speed_kph = speed_mps * 3.6

                # Optional: Add smoothing (e.g., moving average) here if speeds are erratic
                # ...

        return speed_kph


    def process_frame(self, frame):
        """Processes a single frame: tracks objects, estimates speed, performs OCR, saves data."""
        current_time_ts = time() # Get current timestamp for calculations
        current_dt = datetime.now() # Get datetime object for logging
        current_date_str = current_dt.strftime("%Y-%m-%d")
        current_time_str = current_dt.strftime("%H:%M:%S")

        # Initialize Annotator for drawing on the frame
        annotator = Annotator(frame, line_width=2, example=str(self.class_names))

        # Perform object tracking
        try:
            results = self.model.track(frame, persist=True, show=False, verbose=False, conf=CONFIDENCE_THRESHOLD)
        except Exception as e:
            print(f"Error during model tracking: {e}")
            return frame # Return original frame if tracking fails

        # Check if tracking results were obtained
        if results[0].boxes.id is None:
            # print("No tracks detected in this frame.")
            return frame # Return original frame if no tracks

        # Extract tracking data
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        # Process each tracked object
        for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
            # Calculate center point of the bounding box
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Update track history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append((current_time_ts, center_x, center_y))

            # Keep history length manageable (optional, e.g., last 10 points)
            # self.track_history[track_id] = self.track_history[track_id][-10:]

            # Estimate speed
            estimated_speed = self._estimate_speed_for_track(track_id, current_time_ts)
            if estimated_speed is not None:
                self.speed_estimates[track_id] = estimated_speed # Store latest speed

            # Get class name, handle potential errors if cls_id is invalid
            class_name = self.class_names.get(int(cls_id), "Unknown")

            # Prepare label for annotation
            speed_label = f"{int(self.speed_estimates[track_id])} km/h" if track_id in self.speed_estimates else "..."
            label = f"ID: {track_id} {class_name} {speed_label}"

            # Annotate the frame with bounding box and label
            annotator.box_label(box, label=label, color=colors(track_id, True))

            # Perform OCR and Save Data (only if speed is available and not already logged)
            if track_id in self.speed_estimates and track_id not in self.logged_ids:
                # Crop the bounding box area for OCR
                # Add slight padding (optional, can sometimes help OCR)
                pad = 5
                crop_y1 = max(0, y1 - pad)
                crop_y2 = min(frame.shape[0], y2 + pad)
                crop_x1 = max(0, x1 - pad)
                crop_x2 = min(frame.shape[1], x2 + pad)

                if crop_y1 < crop_y2 and crop_x1 < crop_x2: # Ensure valid crop dimensions
                    cropped_image = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    # Perform OCR on the cropped region
                    ocr_text = self.perform_ocr(cropped_image)

                    # Save if OCR text is found
                    if ocr_text:
                        self.save_to_csv(
                            current_date_str,
                            current_time_str,
                            track_id,
                            class_name,
                            self.speed_estimates[track_id], # Current estimated speed
                            ocr_text
                        )
                        self.logged_ids.add(track_id) # Mark as logged for this run
                # else: # Optional: Log if crop was invalid
                     # print(f"Warning: Invalid crop dimensions for ID {track_id}. Skipping OCR.")

        # Return the annotated frame
        return annotator.result()

    def close_csv(self):
        """Closes the CSV file cleanly."""
        if hasattr(self, 'csv_file') and self.csv_file and not self.csv_file.closed:
            try:
                self.csv_file.close()
                print(f"CSV file '{self.csv_filename}' closed.")
            except Exception as e:
                print(f"Error closing CSV file: {e}")


# --- Main Execution ---

if __name__ == "__main__":
    # --- Video Capture Setup ---
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        exit()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        exit()

    # --- Estimator Initialization ---
    speed_estimator = None
    try:
        # Load model once to get class names if not provided manually
        temp_model = YOLO(MODEL_PATH)
        class_names = temp_model.names
        del temp_model # Free memory

        speed_estimator = SpeedEstimator(
            model_path=MODEL_PATH,
            class_names=class_names, # Pass the names from the loaded model
            csv_filename=CSV_FILENAME
        )
    except Exception as e:
        print(f"Error initializing SpeedEstimator or loading model: {e}")
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        exit()

    # --- Frame Processing Loop ---
    frame_count = 0
    print("Starting video processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video file reached.")
            break

        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue # Skip frame

        # Resize frame
        try:
            frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        except Exception as e:
            print(f"Error resizing frame {frame_count}: {e}")
            continue # Skip this frame

        # Process the frame using the SpeedEstimator instance
        try:
            result_frame = speed_estimator.process_frame(frame_resized)
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            # Optionally save the problematic frame for debugging
            # cv2.imwrite(f"error_frame_{frame_count}.png", frame_resized)
            result_frame = frame_resized # Show the original frame on error

        # Display the result
        cv2.imshow("Speed Estimation and OCR", result_frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing stopped by user.")
            break

    # --- Cleanup ---
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    if speed_estimator:
        speed_estimator.close_csv() # Ensure CSV is closed
    print("Processing finished.")