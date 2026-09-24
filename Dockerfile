FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYSPARK_PYTHON=/usr/local/bin/python \
    PATH=/usr/lib/jvm/java-17-openjdk-amd64/bin:$PATH

WORKDIR /opt/app/Lab7DataScience
RUN pip install --no-cache-dir \
    pyspark==3.5.1 pandas==2.2.3 openpyxl==3.1.5 pyarrow==20.0.0 \
    matplotlib==3.10.1 seaborn==0.13.2 notebook==7.4.2 \
    jupyterlab==4.4.2 nbformat==5.10.4 nbconvert==7.16.6 requests==2.32.3

RUN useradd -ms /bin/bash spark && mkdir -p /opt/app/Lab7DataScience && \
    chown -R spark:spark /opt/app
USER spark
EXPOSE 8888
ENTRYPOINT ["tini", "--"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=", "--ServerApp.root_dir=/opt/app/Lab7DataScience"]
