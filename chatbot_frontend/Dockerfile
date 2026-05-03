FROM python:3.11-slim

WORKDIR /app

COPY ./src/ /app

COPY ./pyproject.toml /code/pyproject.toml
RUN pip install --no-cache-dir /code/.

RUN sed -i 's/\r$//' /app/entrypoint.sh

# Streamlit theme config
RUN mkdir -p /root/.streamlit
COPY ./src/.streamlit/config.toml /root/.streamlit/config.toml

CMD ["sh", "/app/entrypoint.sh"]
