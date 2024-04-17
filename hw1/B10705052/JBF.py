import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6 * sigma_s + 1
        self.pad_w = 3 * sigma_s

    def _create_spatial_kernel(self):
        ax = np.arange(-self.pad_w, self.pad_w + 1)
        x, y = np.meshgrid(ax, ax)
        spatial_kernel = -(x**2 + y**2) / (self.sigma_s**2 * 2)
        return spatial_kernel.reshape(-1)

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(
            img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE
        )
        padded_guidance = cv2.copyMakeBorder(
            guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE
        )

        guidance = guidance / 255
        padded_guidance = padded_guidance / 255

        spatial_kernel = self._create_spatial_kernel()
        output = np.zeros(img.shape)
        
        if len(guidance.shape) == 2:
            guidance = guidance[:, :, np.newaxis]
            padded_guidance = padded_guidance[:, :, np.newaxis]
            
        total_weight = np.zeros(img.shape[:2])
        for i in range(0, self.wndw_size * self.wndw_size):
            x = i % self.wndw_size
            y = i // self.wndw_size
            shifted_guidance = padded_guidance[
                y : y + img.shape[0], x : x + img.shape[1]
            ]
            squared_diff = np.sum((shifted_guidance - guidance) ** 2, axis=2)
            weight = np.exp(
                squared_diff / (-2 * self.sigma_r**2) + spatial_kernel[i]
            )
            total_weight += weight

            for c in range(3):
                output[:, :, c] += (
                    weight
                    * padded_img[y : y + img.shape[0], x : x + img.shape[1], c]
                )

        output /= total_weight[:, :, np.newaxis]
        return np.clip(output, 0, 255).astype(np.uint8)
