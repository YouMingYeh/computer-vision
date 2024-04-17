import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter

import csv


def read_settings(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        lines = list(reader)

        # Parse RGB values
        rgb_values = [tuple(map(float, line)) for line in lines[1:-1]]

        # Parse sigma_s and sigma_r
        sigma_s, sigma_r = map(float, lines[-1][1::2])

        return rgb_values, int(sigma_s), sigma_r


def compute_cost(filtered_image, reference_image):
    return np.sum(
        np.abs(filtered_image.astype('int32') - reference_image.astype('int32'))
    )


def save_image(image, path):
    cv2.imwrite(path, image)


def main():
    parser = argparse.ArgumentParser(
        description="main function of joint bilateral filter"
    )
    parser.add_argument(
        "--image_path", default="./testdata/1.png", help="path to input image"
    )
    parser.add_argument(
        "--setting_path",
        default="./testdata/1_setting.txt",
        help="path to setting file",
    )
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ### TODO ###
    rgb_values, sigma_s, sigma_r = read_settings(args.setting_path)

    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)

    costs = []
    jbf = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
    # bf = cv2.cvtColor(bf, cv2.COLOR_RGB2BGR)
    # jbf = cv2.cvtColor(jbf, cv2.COLOR_RGB2BGR)
    cost = compute_cost(jbf, bf)
    costs.append(cost)

    highest_cost = cost
    lowest_cost = cost
    highest_cost_jbf = jbf
    lowest_cost_jbf = jbf
    highest_cost_gray = img_gray
    lowest_cost_gray = img_gray

    for i, rgb in enumerate(rgb_values):
        w_r, w_g, w_b = rgb
        img_converted = (
            w_r * img_rgb[:, :, 0] + w_g * img_rgb[:, :, 1] + w_b * img_rgb[:, :, 2]
        )
        jbf = JBF.joint_bilateral_filter(img_rgb, img_converted).astype(np.uint8)
        # jbf = cv2.cvtColor(jbf, cv2.COLOR_RGB2BGR)
        cost = compute_cost(jbf, bf)
        costs.append(cost)

        if cost > highest_cost:
            highest_cost = cost
            highest_cost_jbf = jbf
            highest_cost_gray = img_converted
        if cost < lowest_cost:
            lowest_cost = cost
            lowest_cost_jbf = jbf
            lowest_cost_gray = img_converted

    print(costs)
    highest_cost_jbf = cv2.cvtColor(highest_cost_jbf, cv2.COLOR_RGB2BGR)
    lowest_cost_jbf = cv2.cvtColor(lowest_cost_jbf, cv2.COLOR_RGB2BGR)
    
    save_image(highest_cost_jbf, "./out/highest_cost_jbf.png")
    save_image(lowest_cost_jbf, "./out/lowest_cost_jbf.png")
    save_image(highest_cost_gray, "./out/highest_cost_gray.png")
    save_image(lowest_cost_gray, "./out/lowest_cost_gray.png")


if __name__ == "__main__":
    main()
