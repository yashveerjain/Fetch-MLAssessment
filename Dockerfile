From pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

COPY ./* ~/home/. 

WORKDIR ~/home

RUN pip install -r requirements.txt

CMD ["python", "results.py"]