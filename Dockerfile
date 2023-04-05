FROM python:3.10

WORKDIR /trailbrake_judge

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

ENV PORT=8080

EXPOSE 8080 2345

CMD [ "python", "app.py" ]