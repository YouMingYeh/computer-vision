import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2 ** (1 / 4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_gaussian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        
        gaussian_images = [[], []]
        dog_images = [[], []]
        # Create Gaussian blurred images for each octave
        for octave in range(self.num_octaves):
            for i in range(self.num_gaussian_images_per_octave):
                sigma = self.sigma ** i
                if i == 0:
                    gaussian_images[octave].append(image)
                    continue
                gaussian_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
                gaussian_images[octave].append(gaussian_image)
                # show the image
            if octave < self.num_octaves - 1:
                image = cv2.resize(gaussian_images[octave][-1], (0, 0), fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)
        # Create DoG images by subtracting consecutive Gaussian blurred images
        for octave in range(self.num_octaves):
            for i in range(self.num_DoG_images_per_octave):
                dog_image = cv2.subtract(gaussian_images[octave][i + 1], gaussian_images[octave][i])
                dog_images[octave].append(dog_image)
                
        # self.save_images(dog_images[0], './out/DoG1-')
        # self.save_images(dog_images[1], './out/DoG2-')
        
        keypoints = self.find_local_extrema(dog_images, self.threshold)

        # Deduplicate keypoints
        keypoints = np.unique(keypoints, axis=0)
        
        keypoints = keypoints[np.lexsort((keypoints[:, 1], keypoints[:, 0]))]
        return keypoints

    def find_local_extrema(self, dog_images, threshold):
        keypoints = []
        for octave, images in enumerate(dog_images):
            for i in range(1, len(images) - 1):
                prev_img, current_img, next_img = images[i - 1], images[i], images[i + 1]
                for y in range(1, current_img.shape[0] - 1):
                    for x in range(1, current_img.shape[1] - 1):
                        block = np.stack([prev_img[y-1:y+2, x-1:x+2], current_img[y-1:y+2, x-1:x+2], next_img[y-1:y+2, x-1:x+2]], axis=-1)
                        if self.is_local_extrema(block, threshold):
                            # Swap the order of x and y when appending to keypoints
                            # Also adjust for octave scaling
                            keypoints.append((y * (2 ** octave), x * (2 ** octave)))
        return np.array(keypoints)


    def is_local_extrema(self, block, threshold):
        center_pixel = block[1, 1, 1]
        if abs(center_pixel) > threshold:
            if center_pixel > 0:
                return center_pixel == np.max(block)
            else:
                return center_pixel == np.min(block)
        return False
    
    def visualize_keypoints(self, image, keypoints):
        for (x, y) in keypoints:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        cv2.imshow('keypoints', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def show_images(self, images, title):
        for i, image in enumerate(images):
            display_image = image.astype(np.uint8)
            cv2.imshow(title + str(i), display_image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def save_images(self, images, title):
        for i, image in enumerate(images):
            display_image = image.astype(np.uint8)
            cv2.imwrite(title + str(i+1) + '.png', display_image)
