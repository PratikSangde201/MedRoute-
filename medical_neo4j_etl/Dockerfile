FROM python:3.11-slim

WORKDIR /app

COPY ./src/ /app

COPY ./pyproject.toml /code/pyproject.toml
RUN pip install --no-cache-dir /code/.

# Strip Windows CRLF so entrypoint works on Linux (common Docker Desktop issue)
RUN sed -i 's/\r$//' /app/entrypoint.sh

CMD ["sh", "/app/entrypoint.sh"]