# File: docker/images/knowledgeBase/Dockerfile
FROM busybox

# Create necessary directory
RUN mkdir -p /knowledgeBase

COPY results/trained_models/ann_training_metrics.csv /knowledgeBase/
COPY results/trained_models/currentAiSolution.keras /knowledgeBase/
COPY results/trained_models/currentOlsSolution.pkl /knowledgeBase/
COPY results/trained_models/ols_model_summary.txt /knowledgeBase/

COPY docker/README.md /knowledgeBase/