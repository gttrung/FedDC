FROM ubuntu:22.04

RUN apt update && apt install -y python3 python3-pip sudo

RUN useradd -m FedDC

RUN chown -R FedDC:FedDC /home/FedDC

COPY --chown=FedDC . /home/FedDC/Exp

USER FedDC

RUN cd /home/FedDC/Exp/ && pip3 install -r requirements.txt 

WORKDIR /home/FedDC/Exp