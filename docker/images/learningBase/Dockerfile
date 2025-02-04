# File: docker/images/learningBase/Dockerfile
FROM busybox

RUN mkdir -p /learningBase/train /learningBase/validation

COPY data/processed/training_data.csv /learningBase/train/
COPY data/processed/test_data.csv /learningBase/validation/

COPY docker/README.md /learningBase/