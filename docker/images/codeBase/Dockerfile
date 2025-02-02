FROM tensorflow/tensorflow

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install matplotlib pandas numpy seaborn scikit-learn statsmodels

# Create necessary directory
RUN mkdir -p /tmp/codeBase

COPY ./code/models/ANN.py /tmp/codeBase/
COPY ./code/models/OLS.py /tmp/codeBase/

COPY docker/README.md /tmp/codeBase/

CMD ["sh", "-c", "echo 'codeBase_student-performance-predictor image is ready.' && tail -f /dev/null"]