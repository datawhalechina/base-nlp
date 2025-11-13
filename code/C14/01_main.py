# main.py

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json

# --- 1. 应用与日志配置 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI 模型API",
    description="一个用于演示如何使用FastAPI部署模型的简单API",
    version="1.0.0",
)

# --- 2. 请求体数据模型 ---
class PredictRequest(BaseModel):
    text: str
    model_name: str | None = None # 可选字段

# --- (模拟)模型加载 ---
# from your_model_module import Model
# model = Model()
# logger.info("模型加载成功")

# --- 3. 预测逻辑路由 ---
@app.post("/predict/")
async def predict(request: PredictRequest):
    """
    接收文本输入，并返回模型预测结果。
    """
    try:
        # 参数过滤与校验
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="输入文本不能为空")

        logger.info(f"接收到请求: {text}")

        # 调用模型获取预测结果 (此处为模拟)
        # prediction = model.predict(text)
        prediction = f"模型对'{text}'的预测结果"
        logger.info(f"模型预测结果: {prediction}")
        
        # 结果转换与输出
        response_data = {
            "code": 0,
            "message": "成功",
            "data": {
                "input_text": text,
                "prediction": prediction
            }
        }
        return response_data

    except Exception as e:
        # 全局异常捕获
        logger.error(f"服务器内部错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

# --- 4. 其他辅助路由 ---
@app.get("/")
async def root():
    return {"message": "欢迎使用 API"}

@app.get("/health")
async def health_check(verbose: bool = False):
    """
    健康检查，可附带详细信息。
    """
    if verbose:
        return {"status": "ok", "details": "All systems operational."}
    return {"status": "ok"}