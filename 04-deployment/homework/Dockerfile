FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "scorer.py", "./" ]

RUN pipenv install --system --deploy

ENTRYPOINT [ "python3", "scorer.py", "2021", "4"]
