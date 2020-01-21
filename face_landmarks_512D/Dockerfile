FROM mxnet/python:nightly_gpu_cu90_mkl_py2

RUN apt-get update
RUN apt-get -y install curl gnupg
RUN curl -sL https://deb.nodesource.com/setup_11.x  | bash -
RUN apt-get -y install nodejs

RUN pip install pika
RUN pip install python-dotenv
RUN npm install -g nodemon
RUN pip install opencv-python
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install numpy scipy scikit-learn
RUN pip install scikit-image

WORKDIR /mxnet
COPY . .