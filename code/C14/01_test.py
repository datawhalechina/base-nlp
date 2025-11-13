from fastapi import FastAPI

# 1. 创建一个 FastAPI 应用实例
app = FastAPI()

# 2. 定义一个路径操作（路由）
# 使用 async def 是 FastAPI 的推荐做法，尤其是在有 I/O 操作时
@app.get("/")
async def read_root():
    # 3. 返回一个字典，FastAPI会自动转换为JSON格式
    return {"Hello": "World"}