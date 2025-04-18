# USAGE: python blip2_server.py

import io
import base64
import torch
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import Blip2Processor, Blip2Model
from torch.utils.data import DataLoader
from PIL import Image
from contextlib import asynccontextmanager


# 配置参数
BATCH_SIZE = 8
MODEL_PATH = '../model/blip2-opt-2.7b'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"


models = {}


# 初始化模型和处理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    processor = Blip2Processor.from_pretrained(MODEL_PATH)
    model = Blip2Model.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    models["processor"] = processor
    models["model"] = model
    yield
    models.clear()


app = FastAPI(lifespan=lifespan)


class ImagePathsRequest(BaseModel):
    base64_images: List[str]
    normalize: bool = True


def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)

    # 转换为 PIL.Image 对象
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, base64_images, processor):
        self.base64_images = base64_images
        self.processor = processor

    def __len__(self):
        return len(self.base64_images)

    def __getitem__(self, idx):
        try:
            image = base64_to_image(self.base64_images[idx])
            return self.processor(images=image, return_tensors="pt")['pixel_values'][0]
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing image {idx}: {str(e)}"
            )


# 批量推理函数
def batch_inference(dataloader, model):
    embeddings = []
    with torch.no_grad():
        for pixel_values in dataloader:
            pixel_values = pixel_values.to(DEVICE)
            outputs = model.get_image_features(pixel_values=pixel_values)
            image_embedding = outputs.pooler_output.cpu()
            embeddings.append(image_embedding)
    return torch.cat(embeddings, dim=0)


@app.post("/embeddings/")
async def get_embeddings(request: ImagePathsRequest):
    # 创建数据加载器
    dataset = ImageDataset(request.base64_images, models["processor"])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 执行推理
    embeddings = batch_inference(dataloader, models["model"])

    # 归一化
    if request.normalize:
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-9)  # 防止除零
        embeddings = embeddings / norms

    return {
        "embeddings": embeddings.tolist()
    }


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8210)
