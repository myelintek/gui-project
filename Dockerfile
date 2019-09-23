FROM tensorflow/tensorflow:1.12.0-gpu
RUN apt-get update ; \
    apt-get install -y git vim
RUN cd / ; \
    git clone -b cnn_tf_v1.12_compatible https://github.com/myelintek/benchmarks.git ; \
    cd benchmarks ; \
    git checkout 5efb0ee68ffe2a9ef41c1a81daa55ad1fe4338a9 ; \
    cd ..
RUN pip install toposort ftputil

COPY code /code
COPY standard-networks /standard-networks
ENV PYTHONPATH "/benchmarks/scripts/tf_cnn_benchmarks/:/code/"
CMD ["/bin/bash"]
