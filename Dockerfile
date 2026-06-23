FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt server.py index.html ./
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p outputs

ENV LUMINA_HOST=0.0.0.0
ENV LUMINA_PORT=8765
EXPOSE 8765

CMD ["python", "server.py"]