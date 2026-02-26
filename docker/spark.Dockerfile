FROM python:3.10-slim

# Install OpenJDK 17 for PySpark
RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

WORKDIR /app
ENV PYTHONPATH=/app/src
COPY requirements.txt .
RUN grep -v -E "torch|torchvision|Pillow" requirements.txt > req_spark.txt && pip install --no-cache-dir -r req_spark.txt
COPY src /app/src
COPY tests /app/tests

CMD ["python"]
