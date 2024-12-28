import numpy as np
import towhee

from config import VECTOR_DIMENSION


def normalize_and_adjust(vector: np.ndarray) -> np.ndarray:
    dimension = VECTOR_DIMENSION
    # Step 1: 归一化向量 (L2 Norm)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    # Step 2: 调整维度
    if len(vector) < dimension:
        # 如果向量长度不足，则补 0
        vector = np.pad(vector, (0, dimension - len(vector)), 'constant')
    elif len(vector) > dimension:
        # 如果向量长度超出，则截断
        vector = vector[:dimension]

    return vector



class ImageModel:
    def __init__(self):
        self.pipe = (
            towhee.pipe.input('url')
            .map('url', 'img',
                towhee.ops.image_decode.cv2())
            .map('img', 'embedding', towhee.ops.image_embedding.timm(model_name='efficientnet_b2'))
            .map('embedding', 'normalized_embedding', normalize_and_adjust)  # 向量归一化
            .output("normalized_embedding")
        )
        # self.pipe = (towhee.dummy_input()
        #              .image_decode()
        #              .image_embedding.timm(model_name='efficientnet_b2_pruned')
        #              .tensor_normalize()
        #              .as_function()
        #              )

    def image_extract_feat(self, img_path):
        feat = self.pipe(img_path).get()[0]
        # if isinstance(feat, _Reason):
        #     raise feat.exception
        return feat


if __name__ == "__main__":
    res = ImageModel().image_extract_feat('https://i1.sinaimg.cn/dy/deco/2013/0329/logo/LOGO_1x.png')
    print(len(res))
    print(res)
