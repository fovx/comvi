from yolov3.utils import image_preprocess, postprocess_boxes
from yolov3.utils import nms

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, Load_Yolo_model
from yolov3.configs import *

def detect_image_with_resize(yolo, image_path, output_path, input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0)):
    # 이미지 읽기
    original_image = cv2.imread(image_path)

    # YOLO로 이미지 검출
    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    pred_bbox = yolo.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size=input_size, score_threshold=0.5)
    bboxes = nms(bboxes, iou_threshold=0.5, method='nms')

    # 바운딩 박스 그리기
    for bbox in bboxes:
        coor = np.array(bbox[:3], dtype=np.int32)
        fontScale = 2
        score = bbox[3]
        class_ind = int(bbox[4])
        import os

	# 파일 경로를 역슬래시로 분할하여 클래스 이름 추출
        class_file_path = "model_data/crack_names.txt"
        txt = open(class_file_path, 'r')
        class_names = txt.read().splitlines()  # 모든 클래스 이름을 읽어옵니다.
        class_name = class_names[class_ind]  # class_ind에 해당하는 클래스 이름을 가져옵니다.



        bbox_color = rectangle_colors[class_ind]
        bbox_thick = int(0.6 * (original_image.shape[0] + original_image.shape[1]) / 1000)

        # 바운딩 박스 그리기
        cv2.rectangle(original_image, (coor[0], coor[1]), (coor[2], coor[3]), bbox_color, bbox_thick)

        # 클래스명과 점수 표시
        label = "{}: {:.2f}".format(class_name, score)
        bboxes_draw = cv2.putText(original_image, label, (coor[0], coor[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, fontScale, bbox_color, bbox_thick)

    # 이미지 저장
    cv2.imwrite(output_path, original_image)

    # 이미지 출력
    if show:
        resized_image = cv2.resize(original_image, (original_image.shape[1] // 4, original_image.shape[0] // 4))
        cv2.imshow("Detection", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


image_path = "C:/Users/Administrator/Desktop/comvi/custom_dataset/train/crack(191).jpg"
video_path   = "./IMAGES/test.mp4"
output_path = "C:/Users/admin/juju/Desktop/DLCV/IMAGES/plate_7_detect.jpg"

# YOLO 모델 불러오기
yolo = Load_Yolo_model()

# 이미지 검출 및 출력
detect_image_with_resize(yolo, image_path, output_path, input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 0), (128, 0, 128)])
