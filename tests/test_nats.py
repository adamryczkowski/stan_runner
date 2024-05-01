from nats import NATS
import signal
from src.stan_runner import *

def test_stream():
    stream = create_stream("nats://localhost:43579", True, "test_stream")
    print(stream)

def list_all_tasks(nats_connect: str):
    nc = NATS()

    async def disconnected_cb():
        print("Got disconnected...")

    async def reconnected_cb():
        print("Got reconnected...")

    nc.connect(nats_connect,
               reconnected_cb=reconnected_cb,
               disconnected_cb=disconnected_cb,
               max_reconnect_attempts=-1)


    nc.jetstream()

def test_alive1():

    async def test():
        ns = await connect_to_nats("localhost:4222", "szakal")

        msg = KeepAliver(ns.jetstream(), "stan.test_alive", timeout=20)
        await msg.remove_all_past_messages()
        await msg.keep_alive()

    asyncio.run(test())

def test_alive2():
    """ This tests for keep-aliver in the presence of its unique ID"""
    async def test():
        ns = await connect_to_nats("localhost:4222", "szakal")

        msg = KeepAliver(ns.jetstream(), "stan.test_alive", timeout=20, unique_id="TEST6")
        await msg.remove_all_past_messages()
        await msg.keep_alive()

    asyncio.run(test())

if __name__ == '__main__':
    # test_stream()
    test_alive2()
    # list_all_tasks("nats://localhost:4222")