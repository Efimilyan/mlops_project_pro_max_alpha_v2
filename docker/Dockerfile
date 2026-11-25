FROM nvcr.io/nvidia/tritonserver:26.11-py3@sha256:cdd351866b6dfbbae6abefee8bab47f2808cd0be33b6fb7825c6f591e5cb11e9

RUN pip install --no-cache-dir faiss-cpu==1.16.0 numpy==1.26.4 \
    && python3 -c "import faiss, numpy; print('faiss', faiss.__version__, 'numpy', numpy.__version__)"
