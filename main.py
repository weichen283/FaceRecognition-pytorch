import cv2
import os
import json
import torch
from torchvision import transforms
from model import Model
import mediapipe as mp
import time
from PIL import Image


def predict(image):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img = data_transform(image)
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = Model()
    model.to(device)

    # load model weights
    weights_path = "./Model.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        res = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(res).numpy()

    cla = class_indict[str(predict_cla)]
    prob = res[predict_cla].numpy()

    if cla == 'my_faces':
        return "weichen", prob
    else:
        return "unknown", prob


cap = cv2.VideoCapture(0)
pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()
   
while True:  
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)
            cv2.rectangle(img, bbox, (127, 255, 0), 2)
            res = predict(Image.fromarray(imgRGB))
            cv2.putText(img, res[0], (bbox[0], bbox[1]-20), cv2.FONT_ITALIC, 1, (127, 255, 0), 2)
            cv2.putText(img, str(int(res[1]*100))+'%', (bbox[0], bbox[1]-50), cv2.FONT_ITALIC, 1, (127, 255, 0), 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_ITALIC, 1, (127, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)




