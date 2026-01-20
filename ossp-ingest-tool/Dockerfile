FROM --platform=$TARGETOS/$TARGETARCH golang:1.23 AS sbomqs-builder
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && apt-get -y install git file
WORKDIR /tmp
RUN git clone https://github.com/shaynakapadia/sbomqs-ossp.git 
WORKDIR /tmp/sbomqs-ossp 
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH go build -o /bin/sbomqs-ossp main.go && mv features.yaml /features.yaml

FROM --platform=$TARGETOS/$TARGETARCH python:3.10
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && apt-get -y install git build-essential sqlite3 flex bison libtool make automake autoconf graphviz libjq-dev

COPY --from=sbomqs-builder /bin/sbomqs-ossp /bin/sbomqs-ossp
COPY --from=sbomqs-builder /features.yaml /features.yaml

WORKDIR /home
RUN pip install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install -r requirements.txt


RUN mkdir ossp
COPY ossp/ ossp/
COPY main.py main.py
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE

ENV FLASK_ENV=production
ENV FLASK_DEBUG=False

RUN pip install .
RUN mkdir -p /data/databases && mv ./ossp/DatabaseSchema04012025.sql /data/databases/schema.sql 


EXPOSE 5000

ENTRYPOINT ["python3", "main.py"]
