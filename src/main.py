import uvicorn
import os
from diskcache import Cache
from fastapi import FastAPI, File, UploadFile
from fastapi.param_functions import Form
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from milvus_helpers import MilvusHelper
from config import TOP_K, UPLOAD_PATH, DEFAULT_TABLE
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

# Mkdir '/tmp/search-images'
if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)
    LOGGER.info(f"mkdir the path:{UPLOAD_PATH}")


@app.get('/data')
def get_img(image_path):
    # Get the image file
    try:
        LOGGER.info(f"Successfully load image: {image_path}")
        return FileResponse(image_path)
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}


@app.get('/progress')
def get_progress():
    # Get the progress of dealing with images
    try:
        cache = Cache('./tmp')
        return f"current: {cache['current']}, total: {cache['total']}"
    except Exception as e:
        LOGGER.error(f"upload image error: {e}")
        return {'status': False, 'msg': e}


class Item(BaseModel):
    Table: Optional[str] = None
    File: str


@app.post('/img/load')
async def load_images(item: Item):
    # Insert all the image under the file path to Milvus/MySQL
    try:
        data = do_load(item.Table, item.File, MODEL, MILVUS_CLI)
        print(data)
        resList = []
        for item in data:
            resList.append({'md5': item["md5"], 'uuid': item["uuid"]})
        print("data", data)
        return {'status': True, 'data': resList}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}


@app.post('/img/upload')
async def upload_images(image: UploadFile = File(None), url: str = None, table_name: str = None):
    # Insert the upload image to Milvus/MySQL
    try:
        # Save the upload image to server.
        if image is not None:
            content = await image.read()
            img_path = os.path.join(UPLOAD_PATH, image.filename)
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
async def train_image_upload(image: UploadFile = File(None)):
    try:
        if image is None:
            return {'status': False, 'msg': 'Image is required'}

        content = await image.read()
        img_path = os.path.join(UPLOAD_PATH, image.filename)
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
        print("delete", img_path)
        os.unlink(img_path)  # 删除上传的图片
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
async def search_images(image: UploadFile = File(...), topk: int = Form(TOP_K), table_name: str = None):
    # Search the upload image in Milvus/MySQL
    try:
        # Save the upload image to server.
        content = await image.read()
        img_path = os.path.join(UPLOAD_PATH, image.filename)
        with open(img_path, "wb+") as f:
            f.write(content)
        f.close()
        res = do_search(table_name, img_path, topk, MODEL, MILVUS_CLI)
        LOGGER.info("Successfully searched similar images!")
        os.unlink(img_path)
        return {'status': True, 'data': res}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}


@app.post('/img/count')
async def count_images(table_name: str = None):
    # Returns the total number of images in the system
    try:
        num = do_count(table_name, MILVUS_CLI)
        LOGGER.info("Successfully count the number of images!")
        return num
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}


@app.post('/img/drop')
async def drop_tables(table_name: str = None):
    # Delete the collection of Milvus and MySQL
    try:
        status = do_drop(table_name, MILVUS_CLI)
        LOGGER.info("Successfully drop tables in Milvus and MySQL!")
        return status
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000)
