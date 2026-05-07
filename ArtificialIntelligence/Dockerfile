#image
FROM python:3.11-slim

# wo soll das laufen? WORKDIR /...

# sachen imstallieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#alles in den container kopieren
COPY . .

#server starten
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]