from __future__ import annotations

import numpy as np
import tritonclient.http as httpclient

MODEL_NAME = "knn_search"
INPUT_NAME = "EMBEDDING"
OUTPUT_NAME = "NEIGHBOR_IDS"
DEFAULT_URL = "localhost:8000"


def connect(url: str = DEFAULT_URL) -> httpclient.InferenceServerClient:
    return httpclient.InferenceServerClient(url=url)


def search(client: httpclient.InferenceServerClient, vec: np.ndarray) -> list[int]:
    vec = np.asarray(vec, dtype=np.float32)
    inp = httpclient.InferInput(INPUT_NAME, list(vec.shape), "FP32")
    inp.set_data_from_numpy(vec)
    out = httpclient.InferRequestedOutput(OUTPUT_NAME)
    result = client.infer(MODEL_NAME, inputs=[inp], outputs=[out])
    return [int(i) for i in result.as_numpy(OUTPUT_NAME)]
