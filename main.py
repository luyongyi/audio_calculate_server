from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import api.audio_param_calculate as audio_param_calculate

app = FastAPI(
    title="FastAPI Service",
    description="FastAPI服务",
    version="1.0.0"
)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.router.include_router(audio_param_calculate.router, prefix="/audio_param_calculate")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有请求头
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
