from nats import NATS
from nats.aio.msg import Msg
import asyncio

def main_loop(nats_connect:str):
    async def run():
        nc = NATS()
        js = nc.jetstream()

        async def disconnected_cb():
            print("Got disconnected...")

        async def reconnected_cb():
            print("Got reconnected...")

        await nc.connect(nats_connect,
                         reconnected_cb=reconnected_cb,
                         disconnected_cb=disconnected_cb,
                         max_reconnect_attempts=-1)

        async def message_handler(msg:Msg):
            print(f"Received msg on subject {msg.subject}: {msg}")
            _, model_hash, task_hash, _ = msg.subject.split(".")
            await nc.publish(f"stan.{model_hash}.{task_hash}.result", b'I can help')

        await js.subscribe("stan.*.*.task", cb=message_handler)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
    loop.run_forever()
    loop.close()


