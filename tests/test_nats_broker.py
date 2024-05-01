import asyncio.exceptions
import time

import nats.errors
import nats.js.errors

from src.stan_runner import *


def test_broker():
    async def test():
        broker = await MessageBroker.Create("localhost:4222", "szakal")
        await broker.the_loop()
    try:
        asyncio.run(test(), debug=True)
    except asyncio.exceptions.CancelledError:
        print("Cancelled error occurred.")


def test2():
    async def test():
        nc = await nats.connect("localhost:4222", token="szakal")
        print(f"Connected to NATS server {nc.connected_url}? {nc.is_connected}")
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

    asyncio.run(test(), debug=True)
    # try:
    #     nc = asyncio.run(nats.connect("localhost:4222", token="szakal"))
    #     js = nc.jetstream()
    #     print(f"Connected to NATS server {nc.connected_url}? {nc.is_connected}")
    #     print("Publishing a message...")
    #     asyncio.run(js.publish("stan_runner", b"Hello World!"))
    #     print("Message published.")
    #     print("Getting the stream info...")
    #     info = asyncio.run(js.stream_info("stan_runner"))
    #     print(info)
    # except nats.errors.TimeoutError:
    #     print("Timeout error occurred. Check the NATS server and the 'stan_runner' stream.")
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    test_broker()
    # test2()
