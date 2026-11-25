import json

import faiss
import numpy as np
import triton_python_backend_utils as pb_utils

K = 5


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        self.dim = int(config["input"][0]["dims"][0])
        index_path = config["parameters"]["index_path"]["string_value"]
        self.index = faiss.read_index(index_path)
        if self.index.d != self.dim:
            raise pb_utils.TritonModelException(
                f"index dim {self.index.d} != config dim {self.dim}"
            )
        if self.index.ntotal < K:
            raise pb_utils.TritonModelException(f"index has only {self.index.ntotal} vectors")
        pb_utils.Logger.log_info(
            f"knn_search: loaded {index_path}: {self.index.ntotal} vectors, dim {self.index.d}"
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            vec = pb_utils.get_input_tensor_by_name(request, "EMBEDDING").as_numpy()
            if vec.shape != (self.dim,):
                responses.append(self._error(f"EMBEDDING must have shape [{self.dim}], got {list(vec.shape)}"))
                continue
            if not np.all(np.isfinite(vec)):
                responses.append(self._error("EMBEDDING contains NaN or Inf"))
                continue
            _, ids = self.index.search(np.ascontiguousarray(vec, dtype=np.float32).reshape(1, -1), K)
            out = pb_utils.Tensor("NEIGHBOR_IDS", ids[0].astype(np.int64))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    @staticmethod
    def _error(message):
        return pb_utils.InferenceResponse(output_tensors=[], error=pb_utils.TritonError(message))
