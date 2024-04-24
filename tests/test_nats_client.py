import asyncio.exceptions
import time

import nats.errors
import nats.js.errors

from stan_runner import *


def test1():
    async def test():
        nc = await nats.connect("localhost:4222", token="szakal")
        print(f"Connected to NATS server {nc.connected_url}? {nc.is_connected}")
        js = nc.jetstream()

    asyncio.run(test(), debug=True)


if __name__ == '__main__':
    test1()
