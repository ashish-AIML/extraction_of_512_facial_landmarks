import pika
import threading
import os
import time
from py_common import tMQ
from dotenv import load_dotenv
import sys
import json
sys.stdout.flush()

import routes

# load env file
load_dotenv()

def callback(ch, method, properties, reqBody):
    reqBody = json.loads(str(reqBody))
    if method.exchange == "events":
        print("[IN-EVENT] ",method.routing_key)
    else:
        print("[IN-REQ] "+method.routing_key)

    reqBody["exchange"] = exchange
    if method.routing_key == "face.extract":
        routes.extractFaceCoordinates(ch, properties, reqBody)

# load rabbitMQ details from env file
exchange = os.getenv("RABBIT_MQ_EXCHANGE_NAME")
pattern = os.getenv("RABBIT_MQ_PATTERN")
queue_name = os.getenv("RABBIT_MQ_QUEUE_NAME")

tMQ.InitBrokerOrDie()
pattern_queue = tMQ.NewQueue(exchange,pattern)
tMQ.GetEvents("events",queue_name)

tMQ.channel.queue_bind(exchange=exchange,queue=queue_name,routing_key="face.extract")

# tMQ.channel.basic_consume(queue="face.extract", on_message_callback=callback, auto_ack=True)
tMQ.channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
tMQ.channel.start_consuming()
