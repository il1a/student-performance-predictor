# File: docker/images/codeBase/Dockerfile
FROM tensorflow/tensorflow

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install pandas statsmodels

# Create necessary directory
RUN mkdir -p /codeBase

COPY code/models/ANN_activation.py /codeBase/
COPY code/models/OLS_activation.py /codeBase/

COPY docker/README.md /codeBase/