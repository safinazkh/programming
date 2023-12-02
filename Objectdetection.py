import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the train image
train_image_path = "img_train.jpg"
train_image = cv2.imread(train_image_path)

# Convert the train image to grayscale
train_gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)

# Use SIFT feature detector and descriptor extractor
sift = cv2.SIFT_create()

# Initialize an empty dictionary to store object features
object_features = {}

# Define the names of the objects
object_names = ["airpods", "controller", "pills", "foundation"]

# Function to select ROIs manually
def select_roi(image, object_name):
    roi = cv2.selectROI(image)
    x, y, w, h = roi
    roi_image = image[y:y+h, x:x+w]
    
    # Detect keypoints and compute descriptors for the ROI
    keypoints, descriptors = sift.detectAndCompute(roi_image, None)

    # Store the features for each object
    object_features[object_name] = {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "bounding_box": (x, y, x+w, y+h)  # Store the bounding box coordinates
    }

# Manually select ROIs for each object
for object_name in object_names:
    print(f"Select ROI for {object_name}. Press any key after selection.")
    select_roi(train_gray, object_name)

# Visualize the ROIs and keypoints on the train image
train_image_with_rois = train_image.copy()

for object_name, features in object_features.items():
    x1, y1, x2, y2 = features["bounding_box"]
    cv2.rectangle(train_image_with_rois, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(train_image_with_rois, object_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(train_image_with_rois, cv2.COLOR_BGR2RGB))
plt.title("Train Image with ROIs")
plt.show()

# Function to match and detect the object in a query image
def match_object(query_image, train_features, object_name):
    # Convert the query image to grayscale
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

    # Use SIFT feature detector and descriptor extractor
    sift = cv2.SIFT_create()
    kp_query, des_query = sift.detectAndCompute(query_gray, None)

    # Use a feature matcher (e.g., FLANN) to find the best matches
    flann_params = dict(algorithm=0, trees=5)
    flann = cv2.FlannBasedMatcher(flann_params, {})
    matches = flann.knnMatch(train_features["descriptors"], des_query, k=2)

    # Ratio test to filter good matches
    good_matches = [m[0] for m in matches if m[0].distance < 0.8 * m[1].distance]

    # If enough good matches are found, proceed with homography
    if len(good_matches) > 10:
        # Extract the matched keypoints
        src_pts = np.float32([train_features["keypoints"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_query[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Use findHomography to get the homography matrix
        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Define the bounding box coordinates for the object in the query image
        h, w = train_gray.shape
        train_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        mapped_corners = cv2.perspectiveTransform(train_corners, homography_matrix)

        # Draw the bounding box on the query image
        for corner in mapped_corners:
            cv2.polylines(query_image, [np.int32(corner)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw feature matching lines
        matching_img = cv2.drawMatchesKnn(train_image, train_features["keypoints"], query_image, kp_query, [good_matches], None, flags=2)
        plt.imshow(cv2.cvtColor(matching_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Feature Matching for {object_name.capitalize()}")
        plt.show()

        return True

    return False

# Example usage for each query image
query_image_paths = ["img_query_1.jpg", "img_query_2.jpg", "img_query_3.jpg"]

for query_image_path in query_image_paths:
    query_image = cv2.imread(query_image_path)
    print(f"Enter the object name ({', '.join(object_names)}) to match in {query_image_path}:")
    object_name = input()
    
    if object_name.lower() in object_names:
        object_found = match_object(query_image, object_features[object_name.lower()], object_name.lower())

        if object_found:
            plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
            plt.title(f"{object_name.capitalize()} Found")
            plt.show()
        else:
            print(f"{object_name.capitalize()} Not Found")
    else:
        print("Invalid object name.")
