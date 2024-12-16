# app.py

import os
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from io import BytesIO
import logging
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
LOG_FOLDER = 'logs'
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_FOLDER, "app.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Configure panorama save folder and allowed extensions
PANORAMA_FOLDER = 'panoramas'
if not os.path.exists(PANORAMA_FOLDER):
    os.makedirs(PANORAMA_FOLDER)
    logger.info(f"Created panorama directory at {PANORAMA_FOLDER}")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Max 500MB upload

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    is_allowed = '.' in filename and \
                 filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    logger.debug(f"File '{filename}' allowed: {is_allowed}")
    return is_allowed

def read_image(file_storage):
    """
    Read an image from a FileStorage object into an OpenCV image (NumPy array).

    Args:
        file_storage (werkzeug.datastructures.FileStorage): The uploaded file.

    Returns:
        np.ndarray: The image in BGR format suitable for OpenCV processing.
    """
    try:
        # Read file content into memory
        file_bytes = file_storage.read()
        # Convert bytes data to NumPy array
        np_arr = np.frombuffer(file_bytes, np.uint8)
        # Decode image from NumPy array
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image.")
            return None
        logger.info(f"Image '{file_storage.filename}' decoded successfully.")
        return img
    except Exception as e:
        logger.exception(f"Exception occurred while reading image '{file_storage.filename}'.")
        return None

def resize_image(img, max_width=1200):
    """Resize image to a maximum width while maintaining aspect ratio."""
    height, width = img.shape[:2]
    if width > max_width:
        scaling_factor = max_width / float(width)
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        logger.info(f"Resized image from ({width}, {height}) to ({img.shape[1]}, {img.shape[0]}).")
    return img

def adjust_exposure(img):
    """Adjust image exposure using histogram equalization."""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    logger.info("Adjusted image exposure.")
    return img_output

def group_images_by_elevation(files):
    """
    Group images by their elevation angle based on filename.

    Args:
        files (list): List of FileStorage objects.

    Returns:
        dict: Dictionary with elevation angles as keys and lists of images as values.
    """
    elevation_groups = defaultdict(list)
    for file in files:
        filename = file.filename
        try:
            # Extract elevation from filename, e.g., 'e30_a0.jpg' -> 30
            base_name = os.path.splitext(filename)[0]
            parts = base_name.split('_')
            elevation_part = parts[0]  # 'e30' or 'e-30'
            elevation = int(elevation_part.replace('e', ''))
            elevation_groups[elevation].append(file)
            logger.debug(f"Grouped '{filename}' under elevation {elevation}°.")
        except (IndexError, ValueError) as e:
            logger.error(f"Filename '{filename}' does not match the required format 'e<elevation>_a<azimuth>.<ext>'. Skipping.")
    return elevation_groups

def stitch_images(images):
    """Stitch a list of OpenCV images into a panorama."""
    logger.info("Initializing OpenCV Stitcher.")
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    
    # Optional: Set stitcher parameters here if needed
    # e.g., stitcher.setPanoConfidenceThresh(1)
    
    logger.info("Starting stitching process.")
    status, pano = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        logger.error(f"Stitching failed with status {status}.")
        return None, status

    logger.info("Stitching completed successfully.")
    return pano, status

@app.route("/")
def hello_world():
    return jsonify({"message": "Hello, Vercel!"})

@app.route('/stitch', methods=['POST'])
def stitch():
    logger.info("Received a request to /stitch endpoint.")

    if 'images' not in request.files:
        logger.warning("No 'images' part in the request.")
        return jsonify({'error': 'No images part in the request'}), 400

    files = request.files.getlist('images')
    logger.info(f"Number of images received: {len(files)}")

    if len(files) < 2:
        logger.warning("Insufficient number of images for stitching.")
        return jsonify({'error': 'At least two images are required for stitching'}), 400

    # Group images by elevation
    elevation_groups = group_images_by_elevation(files)
    logger.info(f"Number of elevation groups: {len(elevation_groups)}")

    if not elevation_groups:
        logger.error("No valid elevation groups found. Ensure filenames follow the 'e<elevation>_a<azimuth>.<ext>' format.")
        return jsonify({'error': 'No valid elevation groups found. Ensure filenames follow the "e<elevation>_a<azimuth>.<ext>" format.'}), 400

    horizontal_panoramas = {}

    try:
        # Phase 1: Horizontal Stitching for each elevation group
        for elevation, group_files in elevation_groups.items():
            logger.info(f"Stitching {len(group_files)} images at elevation {elevation}°.")
            images = []
            for file in group_files:
                img = read_image(file)
                if img is None:
                    logger.error(f"Failed to process image {file.filename}.")
                    return jsonify({'error': f'Failed to process image {file.filename}'}), 400
                
                # Preprocessing steps
                img = resize_image(img, max_width=1200)
                img = adjust_exposure(img)
                
                images.append(img)
            
            if len(images) < 2:
                logger.warning(f"Not enough images to stitch at elevation {elevation}°. Skipping this group.")
                continue

            pano, status = stitch_images(images)
            if pano is None:
                logger.error(f"Stitching failed for elevation {elevation}° with status {status}.")
                return jsonify({'error': f'Stitching failed for elevation {elevation}° with status {status}.'}), 500
            
            horizontal_panoramas[elevation] = pano
            logger.info(f"Stitched horizontal panorama for elevation {elevation}°.")
        
        if not horizontal_panoramas:
            logger.error("No horizontal panoramas were successfully stitched.")
            return jsonify({'error': 'No horizontal panoramas were successfully stitched.'}), 500

        # Phase 2: Vertical Stitching of Horizontal Panoramas
        # Sort elevations for vertical stitching (from highest to lowest)
        sorted_elevations = sorted(horizontal_panoramas.keys(), reverse=True)  # e.g., [90, 60, 30, 0, -30, -60, -90]

        # Resize panoramas to have the same width
        reference_width = horizontal_panoramas[sorted_elevations[0]].shape[1]
        resized_panoramas = []
        for elevation in sorted_elevations:
            pano = horizontal_panoramas[elevation]
            if pano.shape[1] != reference_width:
                scaling_factor = reference_width / pano.shape[1]
                pano = cv2.resize(pano, (reference_width, int(pano.shape[0] * scaling_factor)), interpolation=cv2.INTER_AREA)
                logger.info(f"Resized panorama at elevation {elevation}° to width {reference_width}px for vertical stitching.")
            resized_panoramas.append(pano)

        # Initialize the final panorama with the first horizontal panorama
        final_panorama = resized_panoramas[0]

        # Iterate and stitch vertically
        for pano in resized_panoramas[1:]:
            logger.info("Stitching vertically with the next panorama.")
            # Convert panoramas to grayscale for feature matching
            final_gray = cv2.cvtColor(final_panorama, cv2.COLOR_BGR2GRAY)
            pano_gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)

            # Initialize ORB detector
            orb = cv2.ORB_create()

            # Find the keypoints and descriptors with ORB
            keypoints1, descriptors1 = orb.detectAndCompute(final_gray, None)
            keypoints2, descriptors2 = orb.detectAndCompute(pano_gray, None)

            # Initialize BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match descriptors
            matches = bf.match(descriptors1, descriptors2)

            # Sort them in the order of their distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract location of good matches
            points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find homography
            homography, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

            if homography is None:
                logger.error("Homography could not be computed for vertical stitching.")
                return jsonify({'error': 'Homography could not be computed for vertical stitching.'}), 500

            # Warp the next panorama to align with the final panorama
            height_final, width_final = final_panorama.shape[:2]
            height_pano, width_pano = pano.shape[:2]

            # Define size for the combined image
            combined_height = height_final + height_pano
            combined_width = max(width_final, width_pano)

            # Create a new image with enough space
            combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

            # Place the final panorama on top
            combined_image[0:height_final, 0:width_final] = final_panorama

            # Warp the new panorama and place it below
            warped_pano = cv2.warpPerspective(pano, homography, (combined_width, combined_height))
            combined_image = cv2.addWeighted(combined_image, 1, warped_pano, 1, 0)

            final_panorama = combined_image
            logger.info("Successfully vertically stitched a panorama.")

        if final_panorama is None:
            logger.error("Final panorama is empty after stitching.")
            return jsonify({'error': 'Final panorama is empty after stitching.'}), 500

        # Encode final panorama to JPEG in memory
        success, buffer = cv2.imencode('.jpg', final_panorama)
        if not success:
            logger.error("Failed to encode final panorama image.")
            return jsonify({'error': 'Failed to encode final panorama image.'}), 500

        panorama_bytes = buffer.tobytes()
        logger.info("Final panorama image encoded successfully.")

        # Save panorama to server
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        panorama_filename = f"panorama_{timestamp}.jpg"
        panorama_path = os.path.join(PANORAMA_FOLDER, panorama_filename)

        with open(panorama_path, 'wb') as f:
            f.write(panorama_bytes)
            logger.info(f"Final panorama saved to {panorama_path}")

        # Send the panorama image back
        logger.info("Sending stitched panorama back to client.")
        return send_file(
            BytesIO(panorama_bytes),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=panorama_filename  # Use 'download_name' for Flask >= 2.0
        )

    except Exception as e:
        logger.exception("An unexpected error occurred during the stitching process.")
        return jsonify({'error': 'An unexpected error occurred during the stitching process.'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server.")
    app.run(debug=True, host='0.0.0.0', port=5000)


















# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# import requests
# from io import BytesIO

# app = Flask(__name__)

# @app.route('/api/stitch', methods=['POST'])
# def stitch_images():
#     data = request.get_json()
#     image_urls = data.get('images', [])

#     if len(image_urls) < 2:
#         return jsonify({'success': False, 'message': 'Not enough images to stitch.'}), 400

#     images = []
#     for url in image_urls:
#         response = requests.get(url)
#         img_array = np.frombuffer(response.content, np.uint8)
#         img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#         if img is not None:
#             images.append(img)

#     if len(images) < 2:
#         return jsonify({'success': False, 'message': 'Failed to load images.'}), 400

#     stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
#     status, pano = stitcher.stitch(images)

#     if status != cv2.Stitcher_OK:
#         return jsonify({'success': False, 'message': f'Stitching failed with status {status}.'}), 500

#     # Encode the panorama image to JPEG
#     _, buffer = cv2.imencode('.jpg', pano)
#     pano_bytes = buffer.tobytes()

#     # Upload the stitched image to a storage service (e.g., AWS S3, Cloudinary)
#     # For simplicity, we'll skip this step and assume the stitched image is returned as base64

#     pano_base64 = base64.b64encode(pano_bytes).decode('utf-8')

#     return jsonify({'success': True, 'stitchedImageBase64': pano_base64})

# if __name__ == '__main__':
#     app.run(debug=True)
