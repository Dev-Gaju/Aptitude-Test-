import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn, cv2, torch
from pydantic_settings import BaseSettings
from pydantic import BaseModel

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
model = torch.hub.load('C:/Users/gazur/PycharmProjects/yolov5', 'custom', path="model/best.pt", source='local', force_reload=True)
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

#make prediction form data
def predict(contents):
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    output = model(img)
    return output


# read image and return output
@app.post('/object-detection')
async def predict_pretrain(image: UploadFile = File(...)):
    contents = await image.read()
    detection = predict(contents)
    detection.show()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5001)
