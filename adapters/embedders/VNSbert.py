from core.embedder import BaseEmbedder
from sentence_transformers import SentenceTransformer
from core.logger import logger

import numpy as np

class VNSbert(BaseEmbedder):
    def __init__(self):
        """
        Khởi tạo mô hình keepitreal/vietnamese-sbert, đây là mô hình được benchmark khá cao dựa trên một anh trên cộng đồng Mì AI đánh giá.
        Ở đây, vì việc tính embedding sẽ không tốn quá nhiều, nên ta sẽ đẩy nó lên 'cpu', hoặc sau này sẽ thử nghiệm với 'gpu'
        """
        self.model_name = 'keepitreal/vietnamese-sbert'
        self.mode = 'cpu'

        try:
            self.model = SentenceTransformer(self.model_name, self.mode)
            self.dimension = self.model.get_sentence_embedding_dimension()

            logger.success(f"[Model] Successfully loaded {self.model_name}.")
        except Exception as e:
            logger.error(f"[Model] Cannot load {self.model_name}. Please check again.")
    
    def embed_query(self, query: str) -> np.ndarray:
        logger.info(f"[Embedding] Embedding query {query} by {self.model_name}")

        #TODO: In furture, there will be some functions to rewrite and expand the query for better qualification. But at now, I just write this function in basic

        embeddings = self.model.encode(query)

        return embeddings