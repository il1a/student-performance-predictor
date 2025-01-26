FROM busybox

RUN mkdir -p /tmp/learningBase/train /tmp/learningBase/validation

COPY data/processed/training_data.csv /tmp/learningBase/train/
COPY data/processed/test_data.csv /tmp/learningBase/validation/

COPY docker/images/learningBase/README.md /tmp/learningBase/

CMD ["sh", "-c", "echo 'learningbase_student-performance-predictor image is ready.' && tail -f /dev/null"]