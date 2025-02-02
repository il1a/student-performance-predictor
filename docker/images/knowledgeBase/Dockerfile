FROM busybox

# Create necessary directory
RUN mkdir -p /tmp/knowledgeBase

COPY ann_training_metrics.csv /tmp/knowledgeBase/
COPY currentAiSolution.keras /tmp/knowledgeBase/
COPY currentOlsSolution.pkl /tmp/knowledgeBase/
COPY ols_model_summary.txt /tmp/knowledgeBase/

COPY docker/README.md /tmp/knowledgeBase/

CMD ["sh", "-c", "echo 'knowledgeBase_student-performance-predictor image is ready.' && tail -f /dev/null"]