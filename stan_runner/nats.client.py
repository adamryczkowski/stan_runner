from nats import NATS


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
