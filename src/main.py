import uuid

import uvicorn
import os
from diskcache import Cache
from fastapi import FastAPI, File, UploadFile
from fastapi.param_functions import Form
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse
from milvus_helpers import MilvusHelper
from config import TOP_K, UPLOAD_PATH, DEFAULT_TABLE, DATA_PATH
from encode import ImageModel
from operators import do_load, do_upload, do_search, do_count, do_drop, drop_image
from logs import LOGGER
from pydantic import BaseModel
from typing import Optional
from urllib.request import urlretrieve

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = ImageModel()
MILVUS_CLI = MilvusHelper()
MILVUS_CLI.init_default()

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    LOGGER.info(f"mkdir the path:{DATA_PATH}")

# Mkdir '/data/uploads'
if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)
    LOGGER.info(f"mkdir the path:{UPLOAD_PATH}")


@app.get('/img/download')
def get_img(uuid: str):
    # 查询是否存在
    resList = MILVUS_CLI.client.query(filter=f"uuid == \"{uuid}\"", collection_name=DEFAULT_TABLE, output_fields=["uuid", "meta", "md5"])
    if len(resList) == 0:
        return JSONResponse(status_code=404,content={
            "status": False,
            "msg": "图片不存在"
        })
    imageItem = resList[0]
    uuid = imageItem["uuid"]
    meta = imageItem["meta"]
    if "ext" in meta:
        return JSONResponse(status_code=404,content={
            "status": False,
            "msg": "图片不存在"
        })
    ext = meta["ext"]
    if not os.path.exists(os.path.join(UPLOAD_PATH, f"{uuid}.{ext}")):
        return JSONResponse(status_code=404,content={
            "status": False,
            "msg": "图片不存在"
        })
    return FileResponse(path=os.path.join(UPLOAD_PATH, f"{uuid}.{ext}"), status_code=200)


# @app.get('/progress')
# def get_progress():
#     # Get the progress of dealing with images
#     try:
#         cache = Cache('./tmp')
#         return f"current: {cache['current']}, total: {cache['total']}"
#     except Exception as e:
#         LOGGER.error(f"upload image error: {e}")
#         return {'status': False, 'msg': e}


class Item(BaseModel):
    Table: Optional[str] = None
    File: str


@app.post('/img/upload')
async def upload_images(image: UploadFile = File(None), url: str = None, table_name: str = None):
    # Insert the upload image to Milvus/MySQL
    try:
        # Save the upload image to server.
        if image is not None:
            content = await image.read()
            ext = image.filename.split(".")[-1]
            tempFileName = f"tmp_{uuid.uuid4()}.{ext}"
            img_path = os.path.join(UPLOAD_PATH, tempFileName)
            with open(img_path, "wb+") as f:
                f.write(content)
        elif url is not None:
            img_path = os.path.join(UPLOAD_PATH, os.path.basename(url))
            urlretrieve(url, img_path)
        else:
            return {'status': False, 'msg': 'Image and url are required'}
        resData = do_upload(table_name, img_path, MODEL, MILVUS_CLI)
        return {'status': True, 'data': resData}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}


@app.post('/train/image/upload')
async def train_image_upload(image: UploadFile = File(None), delete: bool = Form(False)):
    try:
        # TODO 缺少meta属性的写入
        if image is None:
            return {'status': False, 'msg': 'Image is required'}

        content = await image.read()
        ext = image.filename.split(".")[-1]
        tempFileName = f"tmp_{uuid.uuid4()}.{ext}"
        img_path = os.path.join(UPLOAD_PATH, tempFileName)
        with open(img_path, "wb+") as f:
            f.write(content)
        f.close()
        trainData = do_upload(DEFAULT_TABLE, img_path, MODEL, MILVUS_CLI)
        if len(trainData) < 1:
            return {'status': False, 'msg': '训练失败'}
        print("trainData", trainData)
        resData = {
            "uuid": trainData[0]["uuid"],
            "md5": trainData[0]["md5"],
            "meta": trainData[0]["meta"]
        }
        if delete:
            os.unlink(img_path)  # 删除上传的图片
            print(f"img_path {img_path}")
        else:
            ext = image.filename.split(".")[-1]
            new_img_path = os.path.join(UPLOAD_PATH, trainData[0]["uuid"] + "." + ext)
            os.rename(img_path, new_img_path)
            print(f"rename {img_path} -> {new_img_path}")
        return {'status': True, 'data': resData}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': "训练异常"}

@app.get('/train/image/delete')
async def train_image_delete(uuid):
    try:
        if uuid is None:
            return {'status': False, 'msg': 'UUID is required'}

        resData = drop_image(DEFAULT_TABLE, uuid=uuid, milvus_cli=MILVUS_CLI)
        return {'status': True, 'data': resData}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': "删除异常"}


@app.post('/img/search')
async def search_images(image: UploadFile = File(...), topk: int = Form(TOP_K)):
    # Search the upload image in Milvus/MySQL
    try:
        # Save the upload image to server.
        content = await image.read()
        ext = image.filename.split(".")[-1]
        tempFileName = f"tmp_{uuid.uuid4()}.{ext}"
        img_path = os.path.join(UPLOAD_PATH, tempFileName)
        with open(img_path, "wb+") as f:
            f.write(content)
        f.close()
        res = do_search(DEFAULT_TABLE, img_path, topk, MODEL, MILVUS_CLI)
        LOGGER.info("Successfully searched similar images!")
        os.unlink(img_path) # 删除上传文件
        return {'status': True, 'data': res}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}


@app.get('/img/count')
async def count_images():
    # Returns the total number of images in the system
    try:
        num = do_count(DEFAULT_TABLE, MILVUS_CLI)
        LOGGER.info("Successfully count the number of images!")
        return {'status': True, 'data': num}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}



if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000)
