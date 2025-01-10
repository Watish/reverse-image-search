import hashlib
import sys
import uuid
from glob import glob
from diskcache import Cache
from config import DEFAULT_TABLE
from logs import LOGGER


def do_upload(table_name, img_path, model, milvus_client, group, extra):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        fileMd5 = get_file_md5(img_path)
        if fileMd5 is not None:
            # 文件存在
            filter = f"md5 == \"{fileMd5}\""
            if group is not None:
                filter += f" and meta[\"group\"] == \"{group}\""
            resList = milvus_client.client.query(filter=filter, collection_name=table_name,
                                                 output_fields=["uuid", "meta", "md5"])
            if len(resList) > 0:
                print(f"MD5 {fileMd5}| 文件存在")
                return resList

        milvus_client.create_collection(table_name)
        feat = model.image_extract_feat(img_path)
        data = milvus_client.insert(table_name, [img_path], [feat], group, extra)
        return data
    except Exception as e:
        LOGGER.error(f"Error with upload : {e}")
        sys.exit(1)


def extract_features(img_dir, model):
    img_list = []
    for path in ['/*.png', '/*.jpg', '/*.jpeg', '/*.PNG', '/*.JPG', '/*.JPEG']:
        img_list.extend(glob(img_dir + path))
    try:
        if len(img_list) == 0:
            raise FileNotFoundError(
                f"There is no image file in {img_dir} and endswith ['/*.png', '/*.jpg', '/*.jpeg', '/*.PNG', '/*.JPG', '/*.JPEG']")
        cache = Cache('./tmp')
        feats = []
        names = []
        total = len(img_list)
        cache['total'] = total
        for i, img_path in enumerate(img_list):
            try:
                norm_feat = model.image_extract_feat(img_path)
                feats.append(norm_feat)
                names.append(img_path)
                cache['current'] = i + 1
                print(f"Extracting feature from image No. {i + 1} , {total} images in total")
            except Exception as e:
                LOGGER.error(f"Error with extracting feature from image:{img_path}, error: {e}")
                continue
        return feats, names
    except Exception as e:
        LOGGER.error(f"Error with extracting feature from image {e}")
        sys.exit(1)


def do_load(table_name, image_dir, model, milvus_client):
    if not table_name:
        table_name = DEFAULT_TABLE
    milvus_client.create_collection(table_name)
    vectors, paths = extract_features(image_dir, model)
    data = milvus_client.insert(table_name, paths, vectors)
    return data


def do_search(table_name, img_path, top_k, model, milvus_client, group):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        feat = model.image_extract_feat(img_path)
        searchData = milvus_client.search_vectors(table_name, [feat], top_k, group)
        return searchData
    except Exception as e:
        LOGGER.error(f"Error with search : {e}")
        sys.exit(1)


def do_count(table_name, milvus_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            return None
        num = milvus_cli.count(table_name)
        return num
    except Exception as e:
        LOGGER.error(f"Error with count table {e}")
        sys.exit(1)


def do_drop(table_name, milvus_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            return f"Milvus doesn't have a collection named {table_name}"
        status = milvus_cli.delete_collection(table_name)
        return status
    except Exception as e:
        LOGGER.error(f"Error with drop table: {e}")
        sys.exit(1)


def drop_image(table_name, uuid: str, milvus_cli, group):
    if not table_name:
        table_name = DEFAULT_TABLE
    return milvus_cli.drop_uuid(collection_name=table_name, uuid=uuid, group=group)


def generate_uuids(length):
    """生成指定长度的 UUID 数组"""
    return [str(uuid.uuid4()) for _ in range(length)]


def get_file_md5(file_path):
    """
    计算文件的 MD5 哈希值
    :param file_path: 文件路径
    :return: 文件的 MD5 哈希值
    """
    md5_hash = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            # 以块的方式读取文件，防止大文件占用过多内存
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
    except FileNotFoundError:
        return None

    return md5_hash.hexdigest()
