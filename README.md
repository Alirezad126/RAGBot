# RAGBot

The assistant behind the chat bubble on
[alirezad126.github.io](https://alirezad126.github.io). It answers questions
about Alireza's background and research by retrieving passages from his CV,
his M.A.Sc. thesis, and a set of written profile notes, then having an
open-weight model answer from them.

## How it is put together

The corpus never changes, which is the fact the whole design turns on. The
index is built once, when the container image is built, and shipped inside it.
So there is no vector database to run, no S3 bucket, and nothing to fetch
before answering a question.

- **Embeddings** run on CPU inside the function, through a `fastembed` ONNX
  model baked into the image. Indexing and retrieval call no API.
- **Generation** goes to Groq, whose free tier covers portfolio traffic.
- **Hosting** is one AWS Lambda function on arm64 behind a function URL, which
  stays inside the perpetual Lambda free tier at this volume. The only standing
  charge is a few cents a month of ECR storage for the image.

Two details are worth knowing. The thesis is 514 of the 565 passages, so a
single top-k over the corpus would bury the CV on exactly the questions
visitors ask first; the biography and the thesis are therefore ranked
separately and both are passed to the model. And `corpus/now.md` is prepended
to every answer regardless of retrieval, because "what are you working on now"
is too common and too important to depend on what scored highest.

## Layout

    app/
      config.py          every knob, all overridable by environment variable
      main.py            FastAPI app: GET /health, POST /chat
      lambda_handler.py  Mangum adapter for the function URL
      index_store.py     loads the baked index
      rag/
        documents.py     corpus to labelled chunks
        chunking.py      recursive character splitting
        retrieval.py     grouped top-k over the baked vectors
        answer.py        prompts, Groq calls, house-style pass
    corpus/              cv.pdf, thesis.pdf, profile.md, now.md
    scripts/build_index.py
    Dockerfile
    deploy.sh

## Deploying

Needs the AWS CLI, Docker, and a Groq API key.

    export GROQ_API_KEY=...
    export ALLOWED_ORIGINS=https://alirezad126.github.io,http://localhost:5173
    ./deploy.sh

It creates the ECR repository, the execution role, the function, and the
function URL on the first run, and updates the function after that. It prints
the function URL at the end, which goes in the site's `VITE_BOT_API_URL`.

## Changing what it knows

Edit `corpus/profile.md`, or drop a new `corpus/cv.pdf`, then run `./deploy.sh`
again. The index is rebuilt during the image build, so there is no separate
indexing step.

## Running it locally

    docker build --platform linux/arm64 --provenance=false --sbom=false -t ragbot .
    docker run --rm -p 9000:8080 -e GROQ_API_KEY=... ragbot

Then POST a function URL event to
`http://localhost:9000/2015-03-31/functions/function/invocations`.
