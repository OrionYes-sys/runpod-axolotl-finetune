FROM winglian/axolotl:main-2.5.0-cu121-2.4.0

# Install runpod SDK for serverless + requests for dataset download
RUN pip install --no-cache-dir runpod pyyaml requests

# Copy handler
WORKDIR /workspace
COPY handler.py /workspace/handler.py

# RunPod serverless entrypoint
CMD ["python", "/workspace/handler.py"]
