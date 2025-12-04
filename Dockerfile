#escape=\

FROM python:3.10.19
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
        bash \
        python3-dev \
        cmake \
        pybind11-dev

COPY ./lsh /app/lsh/
WORKDIR /app/lsh
RUN make .

WORKDIR /app
COPY ./nlsh/requirements.txt /app/nlsh/requirements.txt
RUN pip install --no-cache-dir -r /app/nlsh/requirements.txt
COPY ./nlsh/. .
RUN cp ./lsh/lshlib.so ./nlsh/