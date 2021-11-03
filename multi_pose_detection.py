#!/usr/bin/python

import sys
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

EDGES = {
    (0,1):"m",
    (0,2):"c",
    (1,3):"m",
    (2,4):"c",
    (0,5):"m",
    (0,6):"c",
    (5,7):"m",
    (7,9):"m",
    (6,8):"c",
    (8,10):"c",
    (5,6):"y",
    (5,11):"m",
    (6,12):"c",
    (11,12):"y",
    (11,13):"m",
    (13,15):"m",
    (12,14):"c",
    (14,16):"c"
}
# 图片按比例转换后的最大边（32的倍数）
MAX_SIZE = 256
# 检测点的置信度阈值
CONFIDENCE = 0.1

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints,[y,x,1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame,(int(kx),int(ky)),6,(0,255,0),-1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints,[y, x, 1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),4)

def loop_through_people(frame, keypoints_with_score,
                        edges, confidence_threshold):
    for person in keypoints_with_score:
        draw_connections(frame,person,edges,confidence_threshold)
        draw_keypoints(frame,person,confidence_threshold)

def multi_pose_detection():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model = hub.load("movenet_multipose_lightning_1")
    movenet = model.signatures["serving_default"]

    cap = cv2.VideoCapture("test.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = frame.copy()
        height, width, _ = img.shape
        if height >= width:
            widht_resize = int(round(MAX_SIZE / (float(height) / width) / 32.0) * 32)
            img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0),MAX_SIZE, widht_resize)
        else:
            height_resize = int(round(MAX_SIZE / (float(width) / height) / 32.0) * 32)
            img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), height_resize, MAX_SIZE)
        input_img = tf.cast(img, dtype=tf.int32)
        result = movenet(input_img)
        # 检测最多6个人，每个人有17个关节点，每个关节点有3个值（x，y，置信度）
        # 17个点的顺序是[nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]
        keypoints_with_scores = result["output_0"].numpy()[:,:,:51].reshape((6,17,3))
        loop_through_people(frame, keypoints_with_scores,EDGES, CONFIDENCE)
        cv2.imshow("image", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    multi_pose_detection()
