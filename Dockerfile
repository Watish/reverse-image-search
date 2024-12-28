# 使用镜像代理
FROM dh.noio.top/python:3.12-slim-bullseye

RUN sed -i -E "s/\w+.debian.org/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get remove --purge -y && rm -rf /var/lib/apt/lists/*

# 创建目录
RUN mkdir /app/src/data -p

VOLUME /app/src/data

WORKDIR /app/src
COPY . /app

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN  cd /app/src && \
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
     pip install -r requirements.txt && pip install python-multipart &&  pip cache purge
RUN python3 encode.py

EXPOSE 5000

CMD python3 main.py