from src.stan_runner import *
import nats
import asyncio
import time

async def list_brokers():
    nc = await nats.connect("localhost:4222", token="szakal")
    js = nc.jetstream()
    info = await js.stream_info("stan_runner")
    print("Getting the stream info...")
    print(info)
    try:
        msg = await js.get_last_msg("stan_runner", "stan.broker.alive")
    except nats.js.errors.NotFoundError as e:
        print(f"Stream 'stan_runner' or subject 'stan.broker.alive' not found.")
    else:
        print(f"Got the message: {msg}")

    headers = {"broker_id": "TEST", "timestamp": str(float(time.time()))}
    # await js.publish("stan.broker.alive", b"1", stream=STREAM_NAME, headers=headers)

def test1():
    asyncio.run(list_brokers(), debug=True)

if __name__ == '__main__':
    test1()