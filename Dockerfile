FROM python:3.11-slim

WORKDIR /code

# Install the exact scikit-learn / pandas / numpy versions the model was trained
# with (a pickle is only guaranteed to load under the same scikit-learn version).
COPY models/metadata.json /tmp/metadata.json
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" \
 && pip install --no-cache-dir $(python -c "import json; v = json.load(open('/tmp/metadata.json'))['versions']; print('scikit-learn==%s pandas==%s numpy==%s' % (v['scikit-learn'], v['pandas'], v['numpy']))")

COPY src/ src/
COPY api/ api/
COPY models/ models/

ENV MODEL_DIR=/code/models
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
