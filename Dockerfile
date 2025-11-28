#escape=\

FROM python:3.10.20
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
        bash \
        python3-dev \
        cmake \
        pybind11-dev

COPY ./lsh /app/lsh/
WORKDIR /app/lsh
RUN mkdir -p build \
    && cd build \
    && cmake .. \
    && cmake --build .

WORKDIR /app
COPY ./nlsh/requirements.txt /app/nlsh/requirements.txt
RUN pip install --no-cache-dir -r /app/nlsh/requirements.txt
COPY ./nlsh/. .
RUN cp ./lsh/build/add_module.so ./nlsh/
COPY ./entrypoint.sh /.
ENTRYPOINT [ "/entrypoint.sh" ]