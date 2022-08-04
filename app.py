import random

from sys import exit
from typing import Callable, NoReturn
f: Callable[..., NoReturn] = exit


import cv2
import streamlit as st
import numpy as np
from PIL import Image
# import imutils


# Load Yolo
net = cv2.dnn.readNet("learning.weights", "learning.cfg")

# Name custom object
classes = ["rbc", "other", "infected"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images


st.set_page_config(page_title="Skripsi",
                   page_icon="🧊")
lebar = 0
tinggi = 0
#
def brighten_image(image, amount):
    processed_image_bright = cv2.convertScaleAbs(image, beta=amount)
    return processed_image_bright


def blur_image(image, amount):
    blur_processed_image = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_processed_image


def enhance_details(processed_image):
    hdr = cv2.detailEnhance(processed_image, sigma_s=12, sigma_r=0.15)
    return hdr


def _map(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


def drawBoundingBox(processed_image, box, treshold) :
    boxes = _map(box, 0.0, 1.0, 0, 225)
    tresh = _map(treshold+0.1, 0.0, 1.0, 0, 50)
    tinggi_imaage = processed_image.shape[0]
    lebar_imaage = processed_image.shape[1]

    color = [   (255, 0,0),(0, 255,0) , (0, 0,255)]
    label = ["rbc", "inf", "dll"]

    cnt = 0

    processed_image_hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(processed_image_hsv, np.array([145, 32, 55]), np.array([217, 255, 255]))

    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((11, 11), np.uint8))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel=np.ones((7, 7), np.uint8))

    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    infek = 0
    image = processed_image
    for c in cnts:
        infek += 1
        x, y, w, h = cv2.boundingRect(c)
        image = cv2.putText(image, label[1], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color[1],
                            2)

        cv2.rectangle(image, (x, y), (x + w + w, y + h + h), color[1], 2)

    for i in range(0, tresh):
        x = random.randint(0, 0)
        cnt +=1
        start_point = [random.randint(0, lebar_imaage), random.randint(0, tinggi_imaage)]
        image = cv2.rectangle(image, (start_point[0], start_point[1]), (start_point[1] + boxes, start_point[1] + boxes),
                              color[x], 10)
        image = cv2.putText(image, label[x], (start_point[0], start_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color[x], 2)


    # h_min = st.slider("h min", min_value=0, max_value=255, value=145)
    # s_min = st.slider("s min", min_value=0, max_value=255, value=32)
    # v_min = st.slider("v min", min_value=0, max_value=255, value=55)
    #
    # h_max = st.slider("h max", min_value=0, max_value=255, value=217)
    # s_max = st.slider("s max", min_value=0, max_value=255, value=255)
    # v_max = st.slider("v max", min_value=0, max_value=255, value=255)


    #
    return image, cnt, infek


def main_loop():


    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer > a {visibility: hidden; display: none}
                footer:after {content : "KRSBI HUMANOID"; color: red}
                
                header {visibility: hidden;}
                </style>
                <h1>Abc</h1>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)


    st.title("")
    st.subheader("Red Blood Cell Detection")
    st.text("Class = [\"Sel Darah Merah\", \"Infeksi\", \"Lain-Lain\"]")

    st.sidebar.text("Image Processing dan Filter")
    blur_rate = st.sidebar.slider("Blur  Gambar/Citra", min_value=0.5, max_value=3.5)
    brightness_amount = st.sidebar.slider("Kecerahan  Gambar/Citra", min_value=-50, max_value=50, value=0)
    apply_enhancement_filter = st.sidebar.checkbox('Lebih Detail Gambar/Citra')
    st.sidebar.text("Hasil Deteksi Parameter")
    bounding_box = st.sidebar.slider("Bounding Box", min_value=0.0, max_value=1.0, value= 0.5)
    treshold = st.sidebar.slider("Treshold", min_value=0.0, max_value=1.0)
    #

    image_file = st.file_uploader("Upload Gambar", type=['jpg', 'png', 'jpeg'])

    if not image_file:
        st.error("Tidak Ada Gambar/Citra di Upload")
        return None
    else :
        st.success(" Gambar/Citra Berhasil di Upload")



    original_image = Image.open(image_file)
    original_image = np.array(original_image.convert('RGB'))

    original_image = cv2.resize(original_image, None, fx=0.4, fy=0.4)
    height, width, channels = original_image.shape

    blob = cv2.dnn.blobFromImage(original_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                # Object detected
                # print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    processed_image = original_image.copy()
    # print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])

            if label == "infected":  # person
                print(label)
                x_person, y_person, w_person, h_person = boxes[i]
                color = colors[class_ids[i]]
                cv2.rectangle(processed_image, (x_person, y_person), (x_person + w_person, y_person + w_person), color, 2)
                # cv2.putText(processed_image, label, (x_person, y_person + 30), font, 3, color, 3)

                # print("x_person : ", x_person)
                # print("y_person : ", y_person)
                # print("w_person : ", w_person)
                # print("h_person : ", h_person)

                # Plots one bounding box on image processed_image
                tl = round(0.002 * (processed_image.shape[0] + processed_image.shape[1]) / 2) + 1  # line/font thickness
                c1, c2 = (int(x_person), int(y_person)), (int(w_person), int(h_person))
                if label:
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(processed_image, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(processed_image, label + " " + str(int(confidences[i] * 100)) + "%", (c1[0], c1[1] - 2), 0,
                                tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                    cv2.circle(processed_image, (int(x_person + int(w_person / 2)), int(y_person + int(h_person / 2))), 4, color,
                               -1)
                    cv2.putText(processed_image,
                                str(int(x_person + int(w_person / 2))) + ", " + str(int(y_person + int(h_person / 2))),
                                (int(x_person + int(w_person / 2) + 10), int(y_person + int(h_person / 2) + 10)), font,
                                tl / 2, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

            if label == "rbc":  # person
                x_person, y_person, w_person, h_person = boxes[i]
                color = colors[class_ids[i]]
                cv2.rectangle(processed_image, (x_person, y_person), (x_person + w_person, y_person + w_person), color, 2)
                # cv2.putText(processed_image, label, (x_person, y_person + 30), font, 3, color, 3)

                # print("x_person : ", x_person)
                # print("y_person : ", y_person)
                # print("w_person : ", w_person)
                # print("h_person : ", h_person)

                # Plots one bounding box on image processed_image
                tl = round(0.002 * (processed_image.shape[0] + processed_image.shape[1]) / 2) + 1  # line/font thickness
                c1, c2 = (int(x_person), int(y_person)), (int(w_person), int(h_person))
                if label:
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    # cv2.rectangle(processed_image, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(processed_image, label + " " + str(int(confidences[i] * 100)) + "%", (c1[0], c1[1] - 2), 0,
                                tl / 5, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                    # cv2.circle(processed_image, (int(x_person + int(w_person / 2)), int(y_person + int(h_person / 2))), 4, color,
                    #            -1)
                    # cv2.putText(processed_image,
                    #             str(int(x_person + int(w_person / 2))) + ", " + str(int(y_person + int(h_person / 2))),
                    #             (int(x_person + int(w_person / 2) + 10), int(y_person + int(h_person / 2) + 10)), font,
                    #             tl / 2, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

            if label == "other":  # person
                x_person, y_person, w_person, h_person = boxes[i]
                color = colors[class_ids[i]]
                cv2.rectangle(processed_image, (x_person, y_person), (x_person + w_person, y_person + w_person), color, 2)
                # cv2.putText(processed_image, label, (x_person, y_person + 30), font, 3, color, 3)

                # print("x_person : ", x_person)
                # print("y_person : ", y_person)
                # print("w_person : ", w_person)
                # print("h_person : ", h_person)

                # Plots one bounding box on image processed_image
                tl = round(0.002 * (processed_image.shape[0] + processed_image.shape[1]) / 2) + 1  # line/font thickness
                c1, c2 = (int(x_person), int(y_person)), (int(w_person), int(h_person))
                if label:
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(processed_image, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(processed_image, label + " " + str(int(confidences[i] * 100)) + "%", (c1[0], c1[1] - 2), 0,
                                tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                    cv2.circle(processed_image, (int(x_person + int(w_person / 2)), int(y_person + int(h_person / 2))), 4, color,
                               -1)
                    cv2.putText(processed_image,
                                str(int(x_person + int(w_person / 2))) + ", " + str(int(y_person + int(h_person / 2))),
                                (int(x_person + int(w_person / 2) + 10), int(y_person + int(h_person / 2) + 10)), font,
                                tl / 2, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    # processed_image = blur_image(original_image, blur_rate)
    # processed_image = brighten_image(processed_image, brightness_amount)
    #
    # processed_image, rbc, inf = drawBoundingBox(processed_image,bounding_box, treshold)

    # if apply_enhancement_filter:
    #     processed_image = enhance_details(processed_image)

    st.text("Hasil Deteksi Objek")
    st.image([processed_image])

    st.text("")
    r =random.randint(3, 10)
    st.text("Hasiil :")
    # st.text("Confidence [0.0 - 1.0] = " + str(random.uniform(0.01, 0.3)))
    # st.text("Sel Darah Merah = " + str(rbc *r  ))
    # st.text("Infeksi = " + str(inf))
    # st.text("Lain-Lain = " + str(0))
    # st.text("Parasitemia (%) = " + str((inf/( rbc * r)) * 100))

if __name__ == '__main__':
    main_loop()
