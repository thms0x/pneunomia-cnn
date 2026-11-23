FROM python:3.14-slim

WORKDIR /app
ENV PYTHONPATH=/app/src

# CPU wheels keep the image smaller than the default Linux CUDA build.
RUN pip install --no-cache-dir torch==2.14.1+cpu torchvision==0.29.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi==0.142.2 python-multipart==0.0.32 uvicorn==0.54.0 numpy==2.5.3

RUN useradd -m -u 1000 user
COPY --chown=user src ./src
COPY --chown=user models/model.pt ./models/model.pt

EXPOSE 7860
USER user
CMD ["uvicorn", "pneunomia_cnn.api:app", "--host", "0.0.0.0", "--port", "7860"]
