from nats import NATS
import signal
from stan_runner import *

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

def test_alive():

    async def test():
        ns = await connect_to_nats("localhost:4222", "szakal")

        msg = KeepAliver(ns.jetstream(), "stan.test_alive", timeout=20)
        await msg.remove_all_past_messages()
        await msg.keep_alive()


    # async def shutdown(signal):
    #     print(f"Received exit signal {signal.name}, closing loop...")
    #     loop = asyncio.get_running_loop()
    #     tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    #     [task.cancel() for task in tasks]
    #     await asyncio.gather(*tasks, return_exceptions=True)
    #     loop.stop()
    #
    #
    # loop = asyncio.get_event_loop()
    # loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(shutdown(loop)))
    # loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(shutdown(loop)))

    asyncio.run(test())

if __name__ == '__main__':
    # test_stream()
    test_alive()
    # list_all_tasks("nats://localhost:4222")