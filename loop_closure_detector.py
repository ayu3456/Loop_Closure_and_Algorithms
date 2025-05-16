import cv2
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import urllib.request
import zipfile
from io import BytesIO

class LoopClosureDetector:
    def __init__(self, image_dir="images/loop"):
        """Initialize the Loop Closure Detector"""
        self.image_dir = image_dir
        self.output_dir = "output"
        
        # Create necessary directories
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(nfeatures=1000)
        
        # Initialize FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Parameters for loop closure
        self.similarity_threshold = 0.05  # Very lenient similarity threshold
        self.min_inlier_ratio = 0.1      # Very lenient inlier ratio
        self.min_matches = 5             # Very few required matches
        
        # Storage for frame data
        self.frames = []
        self.descriptors = []
        self.similarity_matrix = None

    def create_distinctive_pattern(self, img, center, color):
        """Create a distinctive pattern at the given center"""
        x, y = center
        size = 40
        # Draw a cross
        cv2.line(img, (x-size, y), (x+size, y), color, 3)
        cv2.line(img, (x, y-size), (x, y+size), color, 3)
        # Draw a circle
        cv2.circle(img, (x, y), size//2, color, 2)
        # Draw a square
        cv2.rectangle(img, (x-size//2, y-size//2), (x+size//2, y+size//2), color, 2)

    def download_sample_images(self):
        """Generate synthetic test images that demonstrate loop closure"""
        # Create a base image with distinctive patterns
        base_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add multiple distinctive patterns
        patterns = [
            ((100, 100), (0, 255, 0)),   # Green pattern top-left
            ((540, 100), (255, 0, 0)),   # Red pattern top-right
            ((100, 380), (0, 0, 255)),   # Blue pattern bottom-left
            ((540, 380), (255, 255, 0)),  # Yellow pattern bottom-right
            ((320, 240), (255, 255, 255)) # White pattern center
        ]
        
        for center, color in patterns:
            self.create_distinctive_pattern(base_img, center, color)
        
        # Add text
        cv2.putText(base_img, "LOOP", (250, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(base_img, "CLOSURE", (230, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Generate a sequence of images with transformations
        num_frames = 10
        for i in range(num_frames):
            save_path = os.path.join(self.image_dir, f"frame_{i+1:03d}.jpg")
            
            if not os.path.exists(save_path):
                # For the last two frames, use transformations similar to first two frames
                if i >= num_frames - 2:
                    base_idx = i - (num_frames - 2)
                    angle = base_idx * 30 / num_frames + 2  # Slightly different angle
                    scale = 1.0 - (base_idx * 0.03) - 0.02  # Slightly different scale
                    tx = base_idx * 10 + 5  # Slightly different translation
                    ty = base_idx * 5 + 5
                else:
                    angle = i * 30 / num_frames
                    scale = 1.0 - (i * 0.03)
                    tx = i * 10
                    ty = i * 5
                
                rows, cols = base_img.shape[:2]
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
                M[0, 2] += tx
                M[1, 2] += ty
                
                # Apply transformation
                transformed = cv2.warpAffine(base_img, M, (cols, rows))
                
                # Add some random noise
                noise = np.random.normal(0, 2, transformed.shape).astype(np.uint8)
                transformed = cv2.add(transformed, noise)
                
                # Save the image
                cv2.imwrite(save_path, transformed)
                print(f"Generated: frame_{i+1:03d}.jpg")
                
        print("Generated test sequence with loop closure")

    def detect_and_compute(self, image):
        """Detect keypoints and compute descriptors using SIFT"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """Match features using FLANN matcher and apply Lowe's ratio test"""
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.8 * n.distance:  # More lenient ratio test
                        good_matches.append(m)
            
            return good_matches
        except Exception as e:
            print(f"Error in feature matching: {e}")
            return []

    def compute_similarity_score(self, matches, num_features1, num_features2):
        """Compute similarity score between two frames based on matches"""
        if not matches or num_features1 == 0 or num_features2 == 0:
            return 0.0
        
        # Normalize by the average number of features
        avg_features = (num_features1 + num_features2) / 2
        similarity = len(matches) / avg_features
        
        return similarity

    def visualize_loop_closure(self, frame_idx1, frame_idx2, matches, mask):
        """Visualize loop closure between two frames"""
        frame1_path = os.path.join(self.image_dir, f"frame_{frame_idx1+1:03d}.jpg")
        frame2_path = os.path.join(self.image_dir, f"frame_{frame_idx2+1:03d}.jpg")
        
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)
        
        if frame1 is None or frame2 is None:
            print(f"Error: Could not load frames for visualization")
            return
        
        # Draw matches
        try:
            matched_img = cv2.drawMatches(
                frame1, self.frames[frame_idx1]['keypoints'],
                frame2, self.frames[frame_idx2]['keypoints'],
                matches, None,
                matchColor=(0, 255, 0),  # Green color for inliers
                singlePointColor=(255, 0, 0),  # Red color for outliers
                matchesMask=mask.ravel().tolist(),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            # Save visualization
            output_path = os.path.join(self.output_dir, f"loop_closure_{frame_idx1+1:03d}_{frame_idx2+1:03d}.jpg")
            cv2.imwrite(output_path, matched_img)
            print(f"Saved visualization to: {output_path}")
            
            # Display
            cv2.imshow('Loop Closure Detection', matched_img)
            cv2.waitKey(1000)  # Display for 1 second
        except Exception as e:
            print(f"Error in visualization: {e}")
            
    def detect_loop_closures(self):
        """Detect loop closures in the sequence"""
        n = len(self.frames)
        self.similarity_matrix = np.zeros((n, n))
        
        # Parameters for loop closure
        self.similarity_threshold = 0.1  # Adjusted threshold
        self.min_inlier_ratio = 0.3     # Adjusted ratio
        self.min_matches = 8            # Adjusted minimum matches
        
        # Compute similarity matrix
        print("\nComputing similarity matrix...")
        for i in range(n):
            for j in range(i + 1, n):
                # Skip consecutive frames
                if j - i <= 2:
                    continue
                    
                matches = self.match_features(self.descriptors[i], self.descriptors[j])
                similarity = self.compute_similarity_score(
                    matches,
                    len(self.frames[i]['keypoints']),
                    len(self.frames[j]['keypoints'])
                )
                
                self.similarity_matrix[i, j] = similarity
                self.similarity_matrix[j, i] = similarity
                
                if similarity > self.similarity_threshold:
                    print(f"High similarity between frames {i+1} and {j+1}: {similarity:.3f}")
        
        # Find loop closures
        print("\nSearching for loop closures...")
        loop_closures = []
        for i in range(n):
            for j in range(i + 3, n):  # Skip immediate neighbors
                if self.similarity_matrix[i, j] > self.similarity_threshold:
                    # Verify loop closure with geometric check
                    matches = self.match_features(self.descriptors[i], self.descriptors[j])
                    if len(matches) > self.min_matches:
                        try:
                            # Get matched keypoints
                            src_pts = np.float32([self.frames[i]['keypoints'][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([self.frames[j]['keypoints'][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                            
                            # Find homography matrix and get inliers mask
                            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            
                            if H is not None:
                                inlier_ratio = np.sum(mask) / len(mask)
                                if inlier_ratio > self.min_inlier_ratio:
                                    loop_closures.append((i, j, self.similarity_matrix[i, j]))
                                    print(f"\nLoop closure detected between frames {i+1} and {j+1}")
                                    print(f"Similarity score: {self.similarity_matrix[i, j]:.3f}")
                                    print(f"Inlier ratio: {inlier_ratio:.3f}")
                                    print(f"Number of matches: {len(matches)}")
                                    
                                    # Visualize loop closure
                                    self.visualize_loop_closure(i, j, matches, mask)
                        except Exception as e:
                            print(f"Error in geometric verification: {e}")
        
        # Visualize similarity matrix
        self.plot_similarity_matrix()
        
        return loop_closures

    def plot_similarity_matrix(self):
        """Plot the similarity matrix"""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.similarity_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Frame Similarity Matrix')
        plt.xlabel('Frame Index')
        plt.ylabel('Frame Index')
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'similarity_matrix.png'))
        plt.close()

    def process_sequence(self):
        """Process the entire sequence of images and detect loop closures"""
        # Get list of images
        image_files = sorted([f for f in os.listdir(self.image_dir) 
                            if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        # Process all frames
        print("Processing frames...")
        for i, image_file in enumerate(image_files):
            frame_path = os.path.join(self.image_dir, image_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"Error: Could not load frame {i}")
                continue
            
            # Detect features
            keypoints, descriptors = self.detect_and_compute(frame)
            
            if descriptors is None:
                print(f"No features detected in frame {i}")
                continue
            
            # Store frame data
            self.frames.append({
                'keypoints': keypoints,
                'path': frame_path
            })
            self.descriptors.append(descriptors)
            
            print(f"Processed frame {i+1}/{len(image_files)}")
        
        # Detect loop closures
        print("\nDetecting loop closures...")
        loop_closures = self.detect_loop_closures()
        
        cv2.destroyAllWindows()
        
        return loop_closures

def main():
    """Main function to run the loop closure detection"""
    detector = LoopClosureDetector()
    
    # Download sample images if they don't exist
    detector.download_sample_images()
    
    # Process the sequence and detect loop closures
    loop_closures = detector.process_sequence()
    
    print("\nLoop Closure Detection Summary:")
    print(f"Total loop closures found: {len(loop_closures)}")
    for i, j, similarity in loop_closures:
        print(f"Loop closure between frames {i+1} and {j+1} (similarity: {similarity:.3f})")

if __name__ == "__main__":
    main() 