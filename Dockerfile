FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# CPU-only torch first (~200 MB instead of ~2.5 GB of CUDA libraries); requirements.txt's
# torch pin is then already satisfied.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==$(grep -E '^torch==' requirements.txt | cut -d= -f3)" \
 && pip install -r requirements.txt \
 && python -m spacy download en_core_web_sm

# Bake the embedding model into the image so the first request doesn't download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps -e .

COPY app ./app
COPY scripts ./scripts
COPY sql ./sql
COPY data/companies.csv data/skills.csv ./data/
COPY data/eval ./data/eval

EXPOSE 8501
CMD ["sh", "-c", "python scripts/init_db.py && streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true"]
