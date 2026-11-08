# arm64 is about 20% cheaper per GB-second than x86 on Lambda, and the ONNX
# embedding runtime has wheels for it.
FROM public.ecr.aws/lambda/python:3.12-arm64

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ${LAMBDA_TASK_ROOT}/app
COPY corpus ${LAMBDA_TASK_ROOT}/corpus
COPY scripts ${LAMBDA_TASK_ROOT}/scripts

# Build the index here, so the embedding model is downloaded and cached in the
# image as well. A cold start then loads it from local disk instead of reaching
# out to Hugging Face, which a Lambda without a NAT route cannot do anyway.
ENV HF_HOME=${LAMBDA_TASK_ROOT}/.cache/huggingface
ENV FASTEMBED_CACHE_PATH=${LAMBDA_TASK_ROOT}/.cache/fastembed
RUN cd ${LAMBDA_TASK_ROOT} && python scripts/build_index.py --corpus corpus --out index

CMD ["app.lambda_handler.handler"]
