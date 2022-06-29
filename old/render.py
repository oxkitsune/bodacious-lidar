from PIL import Image, ImageDraw
import cv2
import numpy as np

def render_magic(inliers, normal):
    # make camera matrix

    print("shape:", inliers.shape)
    rvec = np.zeros(3, np.float)
    tvec = normal / 2
    fx = 200.0
    fy = -200.0
    cx = 1920.0
    cy = 1080.0
    cameraMatrix = np.array([[fx, 0, cx], 
                             [0, fy, cy], 
                             [0,  0,  1]])
    result = cv2.projectPoints(inliers, rvec, tvec, cameraMatrix, None)
    print("result shape:", result[0].shape)
    img = Image.new('RGBA', (3840, 2160), (255, 255, 255, 0))


    print("drawing...")
    for i, vec in enumerate(inliers):
        projected = result[0][i][0]
        pixel = (round(projected[0])), round(projected[1])
        if np.abs(pixel[0]) >= 3840 or np.abs(pixel[1]) >= 2160:
            print("DIDNT SET PIXEL", pixel)
            continue
        # print("setting pixel", pixel)
        values = img.getpixel(pixel)
        img.putpixel(pixel, (255, 0, 0, min(values[3] + 255, 255)))
    print('done, bitch!')

    img.save("out.png")

    # img.save(filename)s
    # for i, vec in enumerate(inliers):
        # i = i + 1
        # print("vec shape at", i, vec.shape)
        # print(vec, "==>", result[0][i])