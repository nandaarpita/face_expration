#start from base
#FROM ubuntu:18.04
FROM alpine
LABEL maintainer="Your Name <youremailaddress@provider.com>"
#RUN apt-get update -y && \
#apt install python3.7
FROM python:3.7
# We copy just the requirements.txt first to leverage Docker cache
COPY ./* /app/
WORKDIR /app
RUN pip install -r requirements.txt
COPY . /app
CMD [ "python", "./flask_web.py" ]
