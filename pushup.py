# Jacob Auman
# The purpose of this program is to count the number of pushups as well as correct user form.

#############################
# Import libraries
import cv2
from ultralytics import YOLO
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import time


####################
# Initialize

m_plot=0
m = 0
b = 0
r_sq = 0

rep_count = 0
frame_count = 0
old_num_peaks = 0

slope_dict = {}
slope_emas_dict = {}


# Initialize the YOLO model
pose_model = YOLO("yolov8s-pose.pt")

keypoint_names = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Create a plot figure 
fig, ax = plt.subplots()
plt.ion()  # Enable interactive mode

####################
# Functions
def linear_regression(keypoint_name_list):
    ############################
    # This function takes the list of L/R body part dictionaries and finds the linear regression for the data
    # Input: list of dictionaries
    # Output: slope and r_squared value from linear regression
    ############################
    
    # Initialize Lists
    x_data = []
    y_data = []

    # Extract x,y coordinates from keypoint dictionaries 
    for keypoint in keypoint_name_list:
        x_data.append(keypoint_dict[keypoint]["x"])
        y_data.append(keypoint_dict[keypoint]["y"])

    N = len(x_data)
    sum_x = sum(x_data)
    sum_y = sum(y_data)
    sum_xy = sum(x*y for x, y in zip(x_data, y_data))
    sum_xx = sum(x*x for x in x_data)
    sum_yy = sum(y*y for y in y_data)

    # Calculate slope
    slope = (N * sum_xy - sum_x * sum_y) / (N * sum_xx - sum_x**2)

    # Calculate the intercept
    intercept = (sum_y - slope * sum_x) / N

    # Calculate R-squared value
    numerator = (N * sum_xy - sum_x * sum_y)**2
    denominator = (N * sum_xx - sum_x**2) * (N * sum_yy - sum_y**2)
    
    # Check for division by zero
    if denominator == 0:
        print("R-squared cannot be calculated because the variance of y is 0")
        r_squared = None

    else:
        r_squared = numerator / denominator

    return slope, intercept , r_squared

def ema_filter(dict_name):
    # This function takes a dictionary of frame numbers and values and applies an exponential moving average filter to the data
    alpha = 0.3 # Smoothing factor 
    ema_value = None
    ema_values = {}

    for frame_number, value in sorted(dict_name.items()):
        if np.isnan(value): # Skip the frame if there's no reading
            ema_values[frame_number] = np.nan
            continue

        if ema_value is None: # Initialize the EMA value for the first valid frame
            ema_value = value
        else: # Compute the EMA
            ema_value = alpha * value + (1 - alpha) * ema_value

        ema_values[frame_number] = ema_value

    return ema_values


####################
# CV Video Capture Settings

# Webcam
# cap = cv2.VideoCapture(0)

# Video
video_name = "pushup3.mp4"

# Open the video file
cap = cv2.VideoCapture(f"videos\{video_name}")
# slow down the video
#cap.set(cv2.CAP_PROP_FPS, 5)

# get dimensions of the original frame
ret , frame = cap.read()
height, width, channels = frame.shape

# Define new height
new_height = height + 100  # increase height by 100 pixels for the bar


####################
# Main Loop
while cap.isOpened():
    frame_count += 1
    success, frame = cap.read()

    if success:
        # Pose detection
        pose_results = pose_model(frame, verbose=False, conf=0.5)

        # Print each body coordinate as a dictionary
        for person in pose_results:
            keypoints = person.keypoints.data[0]
            keypoint_dict = {}
            for keypoint, name in zip(keypoints, keypoint_names):
                x, y, probability = keypoint
                keypoint_dict[name] = {
                    "x": x.item(),
                    "y": y.item(),
                    "probability": probability.item(),
                }
           
        # Check if all of the shoulder, hip, knee, and ankle keypoints can be seen and have a high probability
        probability_threshold = 0.3

        required_keypoints_right = ["right_hip", "right_shoulder"]
        required_keypoints_lower = ["right_ankle", "right_knee"]

        missing_keypoints = [] # Initialize / clear the list

        for keypoint in required_keypoints_right:
            # Check to make sure the keypoint exists and has a high probability
            if keypoint not in keypoint_dict or keypoint_dict[keypoint]["probability"] < probability_threshold:
                missing_keypoints.append(keypoint)

        if missing_keypoints != []:
            print(f"Not all keypoints are visible: {missing_keypoints}") 

            # No m value found for this frame
            null = np.nan
            slope_dict[frame_count] = null 
            slope_emas_dict[frame_count] = null
          
        else:
            if "right_ankle" in keypoint_dict and keypoint_dict["right_ankle"]["probability"] >= probability_threshold:
                # proceed with calculations using the right ankle
                m, b, r_sq = linear_regression(required_keypoints_right + ["right_ankle"])
            elif "right_knee" in keypoint_dict and keypoint_dict["right_knee"]["probability"] >= probability_threshold:
                # proceed with calculations using the right knee
                m, b, r_sq = linear_regression(required_keypoints_right + ["right_knee"])
            else:
                # No m value found for this frame
                null = np.nan
                slope_dict[frame_count] = null 
                slope_emas_dict[frame_count] = null
                print("Neither right ankle nor right knee keypoints are visible with a sufficient probability.")

        m_plot = m # Save the slope value for plotting
        m = abs(m) # Take the absolute value of the slope left / right should not matter

        # Add a filter to remove slope outliers such as standing, sitting, etc.
        slope_tolerance = 0.5
        if m < slope_tolerance and m > -slope_tolerance:
            # Add slope value at fame count to dictionary
            slope_dict[frame_count] = m
            slope_emas_dict[frame_count] = ema_filter(slope_dict)[frame_count]
            #print(f"m: {m}, b: {b}, r_sq: {r_sq}")
        else:
            # No m value found for this frame
            null = np.nan
            slope_dict[frame_count] = null 


        #############################
        # Feature : Rep counting
        # Uses the find_peaks function on the slope ema values to find if the current value is a peak then update the rep counter by 1
        # Dict -> List -> Np Array

        slope_peak_list = list(slope_emas_dict.values())
        slope_peak_array = np.array(slope_peak_list)

        # Find if the current value is a peak or not
        peaks , _ = find_peaks(slope_peak_array, prominence=0.1)
        
        # Add an offset correction number to every value in the peaks list
        peaks = peaks + 3

        num_peaks = len(peaks)

        # If a slope peak found then update the rep counter by 1
        if num_peaks > old_num_peaks:
            rep_count += 1
            print(f"Another Rep!")
    
        # Update the old peak count length
        old_num_peaks = len(peaks)

    


        ############################
        # Display Settings
        sig_fig = 3 # Set a sig fig value for the print statements to reduce size
        m = round(m, sig_fig)
        b = round(b, sig_fig)
        r_sq = round(r_sq, sig_fig)
        
        # Calculate the center point of the frame
        height, width, _ = frame.shape
        center_height = height // 2
        center_width = width // 2


        ############################
        pose_annotated_frame = person.plot()

        ############################
        # CV Display Settings

        # Create a new larger frame
        new_frame = np.zeros((new_height, width, channels), dtype=np.uint8)

        # Insert the original frame with pose annotations into the larger frame
        new_frame[:height, :width] = pose_annotated_frame

        # Draw the bar in the additional space
        cv2.rectangle(new_frame, (0, height), (width, new_height), (33,33,33), -1)


        # Green / Red Square for posture
        x1 = int((width/2)+(0.1*width))
        x2 = int((width/2)-(0.1*width))
    
        if r_sq > 0.8:
            cv2.rectangle(new_frame, (x1, height), (x2, new_height), (12,231,12), -1)
        else:
            cv2.rectangle(new_frame, (x1, height), (x2, new_height), (12,12,231), -1)

           
        # Display the adherance to the line (r_sq)
        text = "Straightness: {}".format(r_sq) 
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = x1 + 20
        text_y = height + 40
        cv2.putText(
            new_frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Display the slop of the back line (m)
        text = "Angle: {}".format(m)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = x1 + 20
        text_y = height + 80
        cv2.putText(
            new_frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Display the slop of the back line (m)
        text = "Rep Counter: {}".format(rep_count)
        text_x = 30
        text_y = int(height + 60)
        cv2.putText(
            new_frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Display the line of best fit over the image of the body and keypoints (y = mx + b)
        null = np.nan
        if m_plot is not null and b is not null:
            if r_sq > 0.8:
                cv2.line(
                    new_frame,
                    (0, int(b)),
                    (width, int(m_plot * width + b)),
                    (0, 192, 0),
                    2,
                )
            else:
                cv2.line(
                    new_frame,
                    (0, int(b)),
                    (width, int(m_plot * width + b)),
                    (0, 0, 255),
                    2,
                )


        # Take the size of the video imput and scale it up on screen
        new_frame = cv2.resize(new_frame, (width * 2, height * 2))

        # Display the frame
        cv2.imshow("Pose Detection", new_frame)


    
        # Press "q" to quit video
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  
    else:
        break


# Release the webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()

# # Wait 1 second
# time.sleep(1)

# # Plot the final graph
# plt.ioff()  # Disable interactive mode
# plt.show()  # Display the final plot
