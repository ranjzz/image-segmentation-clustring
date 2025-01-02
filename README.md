# image-segmentation-clustring
pip install opencv-python
import cv2
import numpy as np
from sklearn.cluster import KMeans

def segment_image(image_path, k=3):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    labels = kmeans.predict(pixels)
    centers = kmeans.cluster_centers_
    
    # Create the segmented image
    segmented_image = centers[labels].reshape(image.shape)
    segmented_image = np.uint8(segmented_image)
    
    return segmented_image

# Example usage
segmented_image = segment_image('path/to/your/image.jpg', k=5)

# Display the original and segmented images
cv2.imshow('Segmented Image', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
