#escape=\

FROM python:3.10.19
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
        bash \
        python3-dev

WORKDIR /app/nlsh
COPY ./nlsh/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

COPY ./lsh /app/lsh/
WORKDIR /app/lsh
RUN make

WORKDIR /app/nlsh
COPY ./nlsh .
RUN cp /app/lsh/lsh .