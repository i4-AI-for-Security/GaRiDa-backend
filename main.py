
import cv2
import uvicorn
#from PIL import Image

from fastapi import File
from fastapi import UploadFile
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
import base64


import os


import inference
import config



# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests
from io import BytesIO

import time
import datetime
from PIL import Image

app = FastAPI()

# # Initialize logging
# my_logger = logging.getLogger()
# my_logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG, filename='logs.log')

detector = None
#masking_img = None

@app.on_event("startup")
def load_facedetector():
    global detector

    # STEP 2: Create an FaceDetector object.
    base_options = python.BaseOptions(model_asset_path= './models/blaze_face_short_range.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

app.add_event_handler("startup", load_facedetector)

@app.post("/api/masking", tags=["masking"])
async def get_masking(file: UploadFile = File(...)):
    start = time.time()
    suffix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    extension = file.filename.split(".")[-1] in ('jpg', 'jpeg', 'png')
    if not extension:
        return f"{extension}: Image must be jpg or png format!!"


    file_path = os.path.join("temp_images", file.filename)
    in_img = './img_file/in_img/in_' + suffix + '.png'
    # 업로드된 이미지를 파일로 저장
    with open(in_img, "wb") as f:
        f.write(file.file.read())

    mp_image = mp.Image.create_from_file(in_img)


    # STEP 4: Detect faces in the input image.
    detection_result = detector.detect(mp_image)
    print(detection_result)

    # Step 5: visualize it & masking it
    image_copy = np.copy(mp_image.numpy_view())
    # visualize
    annotated_image = inference.visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    #masking
    masking_image = inference.masking(image_copy, detection_result)
    masking_ = cv2.cvtColor(masking_image, cv2.COLOR_BGR2RGB)
    remove_masking = inference.remove_grabcut_bg(image_copy, masking_)

    # save image of masking version
    vis_img = './img_file/vis_img/v_'+suffix+'.png'
    masking_img = './img_file/masking_img/mask_'+suffix+'.png'


    cv2.imwrite(masking_img, remove_masking)
    cv2.imwrite(vis_img, rgb_annotated_image)

    headers = {"Content-Disposition": f'attachment; file_keyname = "{suffix}", file_path = {vis_img}'}
    result = {'key_value': suffix, "masking_file_name": masking_img, "visualize_file_name" : vis_img, "time": time.time() - start, 'key_value': suffix}

    # 얼굴 영역 빨간 네모 박스 처리된 이미지 output을 return해야할 거 아냐...
    re_img = cv2.imread(vis_img, cv2.IMREAD_COLOR)
    en_img = np.frombuffer(re_img , dtype=np.uint8)
    result_vis_byte = cv2.imdecode(en_img, cv2.IMREAD_COLOR)


    #result_byte = BytesIO()
    #vis_image = Image.fromarray(rgb_annotated_image)
    #vis_image.save(result_byte, format='PNG',)

    #result_vis_byte = result_byte.getvalue()

    with open(vis_img, 'rb') as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    print(image_base64)

    return {'response': Response(content=image_base64, media_type='image/png'), 'key_value': suffix}




import openai


@ app.post("/api/dalle", tags=["dalle"])
async def dall_face(pp,  key1, key2, key3, key4):
    start = time.time()

    mask_img = './img_file/masking_img/mask_'+pp+'.png'
    in_img = './img_file/in_img/in_'+pp+'.png'



    openai.api_key = "enter the open ai key value"

    #나이 (1-years-old)와 cute, 감정 (happy and bright emotion) 부분 매번 바껴야함
    key1_p, key2_p, key3_p, key4_p = config.KEY1[key1], config.KEY2[key2], config.KEY3[key3], config.KEY4[key4]
    face_prompt = f"""
        a 3D Pixar character of a organized {key1_p} {key2_p} {key4_p} face using
        clay 3D emoji, in front looking at the camera,
        smily, 8k, HD, friendly 3D animated style, {key3_p},
        warn cinematic lighting, ultra hd, realistic, vivid colors, highly detailed, UHD drawing, pen and ink, perfect composition,
        beautiful,8k artistic photography, photorealistic concept art, cartoon, soft natural volumetric cinematic perfect light.
        Removed blur haze, ugly, deformed, distorted, grainy, noisy, blurry, distorted, cropped from Image """

    response = openai.Image.create_edit(
        image= open(in_img, 'rb'),
        mask= open(mask_img, 'rb'),
        prompt=face_prompt,
        n=2,
        size="1024x1024",
    )

    # 이미지 URL 가져오기
    image_url = response.data[0]['url']

    # 이미지 다운로드
    img_response = requests.get(image_url)
    img_data = Image.open(BytesIO(img_response.content))

    # 이미지를 PNG 파일로 저장
    png_name = './img_file/out_img/output_'+pp+'.png'
    img_data.save(png_name, format="PNG")

    result = {'dalle_img_url':image_url, "dalle_img_path":png_name, "time": time.time() - start, 'key_value': pp, 'check':[in_img, mask_img]}
    header = {"Content-Disposition": f'attachment; filename="{png_name}"'}

    with open(png_name, 'rb') as img_file:
        img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
    print(img_base64)

    return Response(content=img_base64, media_type='image/png')


@app.post("/png_img")
async def get_image(image_filepath: str):
    # 이미지 파일의 경로 (예: './images/')
    # 파일을 클라이언트로 반환
    return FileResponse(image_filepath)

@app.post("/", tags=['png_img'])
async def get_image(file_path):
  # 이미지 파일 경로
    return FileResponse(file_path, media_type="image/pnng")

# temp 이미지를 임시로 저장하기 위해
os.makedirs("temp_images", exist_ok=True)



if __name__ == "__main__":
    uvicorn.run("main:app", host = "0.0.0.0", port=8080)

