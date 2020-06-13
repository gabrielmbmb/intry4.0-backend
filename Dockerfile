ARG PYTHON_VERSION=3.8-slim
FROM python:${PYTHON_VERSION}

LABEL maintainer="gmartin_b@usal.es"

# The enviroment variable ensures that the python output is set straight
# to the terminal with out buffering it first
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR=false

RUN apt-get update && \
    # necessary to build psycopg2 package
    apt-get install -y build-essential && \
    apt-get install -y libpq-dev && \
    pip install --upgrade pip && \
    pip install pipenv

ADD Pipfile /tmp
RUN cd /tmp && \
    pipenv lock --requirements > requirements.txt

RUN pip install -r /tmp/requirements.txt

WORKDIR /drf

ADD . .

RUN chmod +x entrypoint.sh

CMD ["/drf/entrypoint.sh"]

