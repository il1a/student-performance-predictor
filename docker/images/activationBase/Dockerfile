# File: docker/images/activationBase/Dockerfile
FROM busybox

# Create necessary directory
RUN mkdir -p /activationBase

COPY data/processed/activation_data.csv /activationBase/

COPY docker/README.md /activationBase/