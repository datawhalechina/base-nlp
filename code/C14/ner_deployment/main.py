import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import NerPredictor  # 使用本地的预测器代码

# --- 全局配置 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "./checkpoints"

# --- 数据模型定义 ---
class NerRequest(BaseModel):
    text: str

# --- FastAPI 应用初始化 ---
app = FastAPI(
    title="命名实体识别 API",
    description="部署 NER 模型",
    version="1.0.0"
)

# --- 模型加载 ---
@app.on_event("startup")
async def startup_event():
    logger.info(f"开始加载模型，来源: {MODEL_DIR}")
    app.state.predictor = NerPredictor(model_dir=MODEL_DIR)
    logger.info("模型加载成功！")


# --- API 路由定义 ---
@app.post("/predict/ner")
async def predict_ner(request: NerRequest):
    """
    接收文本，返回命名实体识别结果。
    """
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="输入文本不能为空")

        logger.info(f"接收到NER请求: '{text}'")
        
        predictor = app.state.predictor
        entities = predictor.predict(text)

        logger.info(f"识别出实体: {entities}")

        return {
            "code": 0,
            "message": "成功",
            "data": {
                "text": text,
                "entities": entities
            }
        }
    except Exception as e:
        logger.error(f"NER预测时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "欢迎使用命名实体识别 (NER) API"}