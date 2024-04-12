FROM supervisely/base-py-sdk:6.73.34

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app
COPY . /app

EXPOSE 80

ENV APP_MODE=production ENV=production

ENTRYPOINT ["python", "-u", "-m", "uvicorn", "src.main:app"]
CMD ["--host", "0.0.0.0", "--port", "80"]
