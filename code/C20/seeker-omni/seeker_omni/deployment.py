import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from PIL import Image
import io

from .inference import SeekerOmniInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置参数
MODEL_CONFIG = "configs/model/base_26m.yaml"
CHECKPOINT_PATH = "checkpoints/seeker-omni"
TEXT_TOKENIZER = "artifacts/tokenizers/bpe_m2chatml_6400"
VISION_MODEL = "google/siglip2-base-patch16-224"
DEVICE = "auto"

# 数据模型
class TextRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"开始加载模型：{CHECKPOINT_PATH}")
    try:
        app.state.inference = SeekerOmniInference(
            model_config=MODEL_CONFIG,
            checkpoint_path=CHECKPOINT_PATH,
            text_tokenizer=TEXT_TOKENIZER,
            vision_model=VISION_MODEL,
            device=DEVICE
        )
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}", exc_info=True)
        raise
    yield
    pass

app = FastAPI(
    title="Seeker Omni 图文多模态模型 API",
    description="支持文本和图像输入的多模态大模型",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/predict/text")
async def predict_text(request: TextRequest):
    """
    接受文本输入，返回模型生成的文本响应。
    """
    try:
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="输入提示不能为空")

        logger.info(f"接收到文本请求：{prompt[:100]}...")

        inference = app.state.inference
        response = inference.generate(
            prompt=prompt,
            image=None,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )

        logger.info(f"生成响应：{response[:100]}...")

        return {
            "code": 0,
            "message": "成功",
            "data": {
                "prompt": prompt,
                "response": response
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文本预测时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


@app.post("/predict/image")
async def predict_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    max_new_tokens: int = Form(512),
    temperature: float = Form(0.7),
    top_p: float = Form(0.95),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.0)
):
    """
    接受文本和图像输入，返回模型生成的文本响应。
    """
    try:
        prompt = prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="输入提示不能为空")

        # 读取图像
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="输入图像不能为空")

        # 验证图像格式
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            raise HTTPException(status_code=400, detail="不支持的图像格式，仅支持 png, jpg, jpeg, webp")

        # 加载图像
        try:
            img = Image.open(io.BytesIO(image_content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无效的图像文件: {e}")

        logger.info(f"接收到图像请求，提示：{prompt[:100]}...")

        inference = app.state.inference
        response = inference.generate(
            prompt=prompt,
            image=img,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )

        logger.info(f"生成响应：{response[:100]}...")

        return {
            "code": 0,
            "message": "成功",
            "data": {
                "prompt": prompt,
                "response": response
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图像预测时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "欢迎使用 Seeker Omni 图文多模态模型 API"}
