import os
import sys
from config import VECTOR_DIMENSION, METRIC_TYPE, DEFAULT_TABLE
from pymilvus import DataType, MilvusClient
from logs import LOGGER
from operators import generate_uuids, get_file_md5


class MilvusHelper:
    """
    MilvusHelper class to manager the Milvus Collection.

    Args:
        host (`str`):
            Milvus server Host.
        port (`str|int`):
            Milvus server port.
        ...
    """

    def __init__(self):
        try:
            self.collection = None
            # 判断目录是否存在
            if not os.path.exists("./data"):
                # 如果不存在则创建
                os.makedirs("./data")
            self.client = MilvusClient("./data/milvus_data.db")
            # connections.connect(host=host, port=port)
            # LOGGER.debug(f"Successfully connect to Milvus with IP:{MILVUS_HOST} and PORT:{MILVUS_PORT}")
        except Exception as e:
            LOGGER.error(f"Failed: {e}")
            sys.exit(1)

    def init_default(self):
        if not self.has_collection(collection_name=DEFAULT_TABLE):
            self.create_collection(collection_name=DEFAULT_TABLE)
        self.client.load_collection(collection_name=DEFAULT_TABLE)

    def set_collection(self, collection_name):
        try:
            self.client.create_collection(collection_name, dimension=VECTOR_DIMENSION)
        except Exception as e:
            LOGGER.error(f"Failed to load data to Milvus: {e}")
            sys.exit(1)

    def has_collection(self, collection_name):
        # Return if Milvus has the collection
        try:
            return self.client.has_collection(collection_name)
            # return utility.has_collection(collection_name)
        except Exception as e:
            LOGGER.error(f"Failed to load data to Milvus: {e}")
            sys.exit(1)

    def create_collection(self, collection_name):
        try:
            if not self.client.has_collection(collection_name):
                # 定义字段
                schema = MilvusClient.create_schema()
                schema.add_field(
                    field_name='id',
                    datatype=DataType.INT64,
                    description='Image ID',
                    max_length=64,
                    is_primary=True,  # 主键字段
                    auto_id=True
                )
                schema.add_field(
                    field_name='uuid',
                    datatype=int(DataType.VARCHAR),
                    description='Image UUID',
                    max_length=64,
                )
                schema.add_field(
                    field_name='md5',
                    datatype=int(DataType.VARCHAR),
                    description='Image MD5',
                    max_length=32,
                )
                schema.add_field(
                    field_name='embedding',
                    datatype=int(DataType.FLOAT_VECTOR),
                    description='Image embedding vectors',
                    dim=VECTOR_DIMENSION  # 确保与实际数据维度一致
                )
                schema.add_field(
                    field_name='meta',
                    datatype=DataType.JSON,
                    description='Image Meta JSON'
                )

                # 创建集合
                self.client.create_collection(collection_name, schema=schema)
                LOGGER.debug(f"Created Milvus collection: {collection_name}")
                self.create_index(collection_name)
            # else:
            #     self.set_collection(collection_name)
            return "OK"
        except Exception as e:
            LOGGER.error(f"Failed to load data to Milvus: {e}")
            sys.exit(1)

    def insert(self, collection_name, path, vectors):
        # Batch insert vectors to milvus collection
        try:
            uuids = generate_uuids(len(path))
            # 将 uuid 添加到 data
            rows = []
            for index, _ in enumerate(vectors):
                # 计算md5
                md5 = get_file_md5(path[index])
                if md5 is None:
                    md5 = ""
                row = {
                    "meta": {
                        "path": path[index]
                    },
                    "embedding": vectors[index],
                    "uuid": uuids[index],
                    "md5": md5
                }

                alreadyExists = False

                if md5 != "":
                    targetRes = self.client.query(filter=f"md5 == \"{md5}\"", collection_name=collection_name, limit=1)
                    if len(targetRes) > 0:
                        print("文件已存在", path[index], md5)
                        row["uuid"] = targetRes[0]["uuid"]
                        row["meta"] = targetRes[0]["meta"]
                        alreadyExists = True

                print("row", row)
                if not alreadyExists:
                    self.client.insert(collection_name, row)
                rows.append(row)
            self.client.load_collection(collection_name)
            LOGGER.debug(
                f"Insert vectors to Milvus in collection: {collection_name} with {len(vectors)} rows")
            return rows
        except Exception as e:
            LOGGER.error(f"Failed to load data to Milvus: {e}")
            sys.exit(1)

    def create_index(self, collection_name):
        try:
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(field_name="embedding", index_type="HNSW", index_name="embedding_index",
                                   metric_type=METRIC_TYPE, params={"nlist": 16384})
            self.client.create_index(collection_name, index_params=index_params)
        except Exception as e:
            LOGGER.error(f"Failed to create index: {e}")
            sys.exit(1)

    def delete_collection(self, collection_name):
        try:
            self.client.drop_collection(collection_name)
            LOGGER.debug("Successfully drop collection!")
            return "ok"
        except Exception as e:
            LOGGER.error(f"Failed to drop collection: {e}")
            sys.exit(1)

    def search_vectors(self, collection_name, vectors, top_k):
        LOGGER.debug(f"Vectors for search: {vectors}")
        try:
            search_params = {"metric_type": METRIC_TYPE, "params": {"nprobe": 16}}
            res = self.client.search(collection_name, data=vectors, anns_field="embedding", search_params=search_params
                                     , limit=top_k, output_fields=["uuid"])
            LOGGER.debug(f"Successfully search in collection: {res}")
            return res
        except Exception as e:
            LOGGER.error(f"Failed to search vectors in Milvus: {e}")
            sys.exit(1)

    def drop_uuid(self, collection_name, uuid):
        return self.client.delete(collection_name=collection_name,
                                  filter=f"uuid == \"{uuid}\"")

    def count(self, collection_name):
        try:
            num = len(self.client.get_collection_stats(collection_name))
            LOGGER.debug(f"Successfully get the num:{num} of the collection:{collection_name}")
            return num
        except Exception as e:
            LOGGER.error(f"Failed to count vectors in Milvus: {e}")
            sys.exit(1)
