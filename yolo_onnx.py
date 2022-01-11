import numpy as np
import cv2
import time

path_model = "weights/"
model_name = "best.onnx"

model = cv2.dnn.readNet(path_model + model_name)
img = cv2.imread('img.jpg')
Width, Height , channels = img.shape
img = cv2.resize(img, (640, 640))
blob = cv2.dnn.blobFromImage(image=img, scalefactor=0.00392, size=(640, 640), mean=(0, 0, 0), swapRB=True)
model.setInput(blob)
def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

output = model.forward(get_output_layers(model))
print(output)
with open('weights/classes.txt', 'r') as f:
   image_net_names = f.read().split('\n')
class_names = [name.split(',')[0] for name in image_net_names]
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.2
nms_threshold = 0.2

for out in output[0]:
    for detection in out:

        scores = detection[4]
        class_id = np.argmax(scores)
        confidence = detection[4]
        if confidence > 0.2:
            center_x = int(detection[0])
            center_y = int(detection[1])
            w = int(detection[2])
            h = int(detection[3])
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = "void"

    color = (255,0,0)

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

for i in indices:

    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    print(confidences[i], round(x), round(y), round(x + w), round(y + h))
    draw_bounding_box(img, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

cv2.imshow("object detection", img)

# wait until any key is pressed
cv2.waitKey()
