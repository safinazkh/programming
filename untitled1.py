import numpy as np
import cv2
import matplotlib.pyplot as plt

# Feature Detection and Description
def detect_and_describe(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# Feature Matching
def match_keypoints(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match des1 and des2
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    
    return good_matches

# Homography Estimation
def estimate_homography(kp1, kp2, des1, des2):
    # Match keypoints and descriptors between des1 and des2
    good_matches = match_keypoints(des1, des2)
    
    if not good_matches:
        print("No valid matches found between the images.")
        return None

    src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Use RANSAC to find the homography matrix
    homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return homography

# Image Warping and Blending
def warp_and_blend_images(img1, img2, homography):
    # Warp the second image to align with the first image
    result = cv2.warpPerspective(img1, homography, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    return result

# Read images
img3 = cv2.imread('image3.jpg')
img2 = cv2.imread('image4.jpg')
img1 = cv2.imread('image5.jpg')

# Feature Detection and Description
kp1, des1 = detect_and_describe(img1)
kp2, des2 = detect_and_describe(img2)
kp3, des3 = detect_and_describe(img3)

# Feature Matching and Blending for Image 1 and Image 2
good_matches1_2 = match_keypoints(des1, des2)
homography1_2 = estimate_homography(kp1, kp2, des1, des2)
result1_2 = warp_and_blend_images(img1, img2, homography1_2)

# Feature Matching and Blending for Result of Image 1 and 2 and Image 3
kp_result, des_result = detect_and_describe(result1_2)
good_matches_result_3 = match_keypoints(des_result, des3)
homography_result_3 = estimate_homography(kp_result, kp3, des_result, des3)
final_result = warp_and_blend_images(result1_2, img3, homography_result_3)

# Display the final result
plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
plt.show()

'''with plot '''
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Feature Detection and Description
def detect_and_describe(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# Feature Matching
def match_keypoints(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=60)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)
    return good_matches

# Homography Estimation
def estimate_homography(kp1, kp2, good_matches):
    src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return homography

# Image Warping and Blending
def warp_and_blend_images(img1, img2, homography):
    result = cv2.warpPerspective(img1, homography, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    return result

# Trim zero pixels from the border of the image
def trim_zeros(image):
    non_zero_rows = np.any(image, axis=1)
    non_zero_cols = np.any(image, axis=0)

    row_start, row_end = np.where(non_zero_rows)[0][[0, -1]]
    col_start, col_end = np.where(non_zero_cols)[0][[0, -1]]

    result = image[row_start:row_end + 1, col_start:col_end + 1]

    return result

# Annotate keypoints on the image with matching points and arrows
def annotate_keypoints(image1, keypoints1, image2, keypoints2, matches, color=(0, 255, 0)):
    annotated_image = np.concatenate((image1, image2), axis=1)
    offset = image1.shape[1]  # Width of the first image

    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx

        if 0 <= query_idx < len(keypoints1) and 0 <= train_idx < len(keypoints2):
            kp1 = keypoints1[query_idx].pt
            kp2 = keypoints2[train_idx].pt
            pt1 = (int(kp1[0]), int(kp1[1]))
            pt2 = (int(kp2[0] + offset), int(kp2[1]))

            cv2.circle(annotated_image, pt1, 5, color, -1)
            cv2.circle(annotated_image, pt2, 5, color, -1)
            cv2.line(annotated_image, pt1, pt2, color, 2)

    return annotated_image

# Read images
img3 = cv2.imread('image3.jpg')
img2 = cv2.imread('image4.jpg')
img1 = cv2.imread('image5.jpg')

# Feature Detection and Description
kp1, des1 = detect_and_describe(img1)
kp2, des2 = detect_and_describe(img2)
kp3, des3 = detect_and_describe(img3)

# Feature Matching and Blending for Image 1 and Image 2
good_matches1_2 = match_keypoints(des1, des2)
homography1_2 = estimate_homography(kp1, kp2, good_matches1_2)
result1_2 = warp_and_blend_images(img1, img2, homography1_2)

# Feature Matching and Blending for Result of Image 1 and 2 and Image 3
kp_result, des_result = detect_and_describe(result1_2)
good_matches_result_3 = match_keypoints(des_result, des3)
homography_result_3 = estimate_homography(kp_result, kp3, good_matches_result_3)
final_result = warp_and_blend_images(result1_2, img3, homography_result_3)

# Trim zero pixels from the final result
final_result_trimmed = trim_zeros(final_result)

# Annotate keypoints on the images with matching points and arrows
annotated_img1 = annotate_keypoints(img1, kp1, img2, kp2, good_matches1_2)
annotated_img2 = annotate_keypoints(img2, kp2, img3, kp3, good_matches_result_3)
annotated_img3 = annotate_keypoints(final_result_trimmed, kp_result, final_result_trimmed, kp_result, good_matches_result_3)
annotated_result1_2 = annotate_keypoints(result1_2, kp_result, result1_2, kp_result, good_matches_result_3)
annotated_final_result = annotate_keypoints(final_result_trimmed, kp_result, final_result_trimmed, kp_result, good_matches_result_3)

# Display the annotated results
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(cv2.cvtColor(annotated_img1, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Image 1 with Keypoints and Matches')

axs[0, 1].imshow(cv2.cvtColor(annotated_img2, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Image 2 with Keypoints and Matches')

axs[0, 2].imshow(cv2.cvtColor(annotated_img3, cv2.COLOR_BGR2RGB))
axs[0, 2].set_title('Image 3 with Keypoints and Matches')

axs[1, 0].imshow(cv2.cvtColor(annotated_result1_2, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Result of Image 1 and 2 with Keypoints and Matches')

axs[1, 1].imshow(cv2.cvtColor(annotated_final_result, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Final Result with Keypoints and Matches')

# Hide empty subplot
axs[1, 2].axis('off')

plt.show()
