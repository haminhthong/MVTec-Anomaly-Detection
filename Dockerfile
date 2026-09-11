FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch-cache

WORKDIR /app

COPY requirements.txt .

# Dùng đúng bản CPU PyTorch giống workflow CI.
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch==2.7.1+cpu" \
        "torchvision==0.22.1+cpu" \
    && grep -Ev '^(torch|torchvision)==' requirements.txt \
        > /tmp/requirements-no-torch.txt \
    && python -m pip install -r /tmp/requirements-no-torch.txt \
    && python -m pip check

# Tải sẵn trọng số backbone để API không cần gọi mạng khi khởi động.
RUN python -c \
    "from torchvision.models import ResNet18_Weights, resnet18; \
     resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"

RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p /app/models \
    && chown -R app:app /app /opt/torch-cache

# Container chỉ nhận source serving; dataset và model được cung cấp bên ngoài.
COPY --chown=app:app src ./src

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c \
      "import urllib.request; \
       urllib.request.urlopen('http://127.0.0.1:8000/live', timeout=3)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
