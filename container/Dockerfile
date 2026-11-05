# Docker equivalent of al.def:
#   docker build -t spinel-al container/
FROM nvcr.io/nvidia/pytorch:25.10-py3
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir --root-user-action=ignore -r /opt/requirements.txt
WORKDIR /work
