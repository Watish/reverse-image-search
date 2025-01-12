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
        self.imageTextPipe = (
            towhee.pipe.input('url')
            .map('url', 'img', towhee.ops.image_decode.cv2_rgb())
            .map('img', 'vec', towhee.ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='image'))
            .output('vec')
        )
        self.imageTextPipe = (
            towhee.pipe.input('text')
            .map('text', 'vec', towhee.ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='text'))
            .output('vec')
        )
        self.image2TextPipe = (
            towhee.pipe.input('url')
            .map('url', 'img', towhee.ops.image_decode.cv2_rgb())
            .map('img', 'text', towhee.ops.image_captioning.clip_caption_reward(model_name='clipRN50_clips_grammar'))
            .output('text')
        )

    def image_extract_feat(self, img_path):
        feat = self.pipe(img_path).get()[0]
        return feat

    def image_to_text(self, img_path):
        feat = self.image2TextPipe(img_path).get()[0]
        return feat

    def image_text_extract_feat(self, img_path):
        feat = self.imageTextPipe(img_path).get()[0]
        return feat

if __name__ == "__main__":
    imagePath = 'https://cross-java-images.oss-cn-zhangjiakou.aliyuncs.com/lglv998/abe83fc4cc91c91538e803bbf37cc886.jpg'
    res = ImageModel().image_extract_feat(img_path=imagePath)
    print(res)
    res = ImageModel().image_to_text(img_path=imagePath)
    print(res)
    res = ImageModel().image_text_extract_feat(img_path=imagePath)
    print(res)
