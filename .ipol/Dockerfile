FROM registry.ipol.im/ipol:v2-py3.11-pytorch

WORKDIR /demo

COPY requirements.txt /demo/requirements.txt
RUN pip install --no-cache-dir -r /demo/requirements.txt

COPY src /demo/src
COPY models /demo/models
