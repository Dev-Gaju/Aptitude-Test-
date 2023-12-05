import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn, cv2, torch
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.responses import JSONResponse, StreamingResponse
from io import BytesIO


# response when run API
class AppConfig(BaseSettings):
    app_name: str = "Object Detection Model Serving Demo"
    version: str = "0.1"


class IndexResponse(BaseModel):
    response: str = "Model Serving Demo"


# Control Version
class VersionResponse(BaseModel):
    app_name: str
    version: str


appconfig = AppConfig()
app = FastAPI(title=appconfig.app_name,
              version=appconfig.version,
              description="A demo backend app for serving object Detection model with FastApi.")

# model import and weights then  process
device = torch.device('cpu')
model = torch.hub.load('C:/Users/gazur/PycharmProjects/yolov5', 'custom', path="model/best.pt", source='local',
                       force_reload=True)
model.to(device)


@app.get("/")
async def index():
    return {"response": appconfig.app_name}


@app.get("/version", response_model=VersionResponse)
def version():
    return {
        "app_name": appconfig.app_name,
        "version": appconfig.version
    }


# make prediction form data
def predict(contents):
    # nparr = np.frombuffer(contents, np.uint8)
    # contents_bytes = bytes(contents)
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img.shape[2] == 1:  # adding updated version
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    output = model(img)
    return img, output


# read image and return output
# @app.post('/object-detection')
# async def predict_pretrain(image: UploadFile = File(...)):
#     contents = await image.read()
#     detection = predict(contents)
#     detection.show()

""" New version of this code"""


@app.post('/object-detection')
async def predict_pretrain(image: UploadFile = File(...)):
    contents = await image.read()
    try:
        """ Return with Bounding box"""
        # visualized detection on the images
        img, detection = predict(contents)
        for box in detection.xyxy[0].cpu().numpy():
            x_min, y_min, x_max, y_max, confidence, class_idx= box
            cv2.rectangle(img, (int(x_min), int(y_min), int(x_max), int(y_max)), (0, 255, 0), 2)
        # convert the images into bytes
        _, img_bytes = cv2.imencode('.png', img)
        img_stream = BytesIO(img_bytes.tobytes())
        return StreamingResponse(content=img_stream, media_type="image/png", )
    except Exception as e:
        return  JSONResponse( content={"error":str(e)}, status_code=500)

    """ just return response """
    #     detection = predict(contents)
    #     """[x_min, y_min, x_max, y_max, confidence, class]"""
    #     return JSONResponse(content=detection.xyxy[0].cpu().numpy().tolist(), media_type="application/json")
    # except Exception as e:
    #     return JSONResponse(content={"error": str(e)}, status_code=500)





if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5001)
