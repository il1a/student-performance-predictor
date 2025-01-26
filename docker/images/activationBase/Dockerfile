FROM busybox

# Create necessary directory
RUN mkdir -p /tmp/activationBase

COPY data/processed/activation_data.csv /tmp/activationBase/
COPY docker/images/activationBase/README.md /tmp/activationBase/

CMD ["sh", "-c", "echo 'activationBase_student-performance-predictor image is ready.' && tail -f /dev/null"]