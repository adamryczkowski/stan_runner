# This is a general unit test for the nats system.
#
# It runs a NATS server in a separate thread.

from asyncio.subprocess import Process

from nats.js.errors import NotFoundError

from stan_runner import *


async def run_the_server() -> Process:
    # Server is run with a command `nats-server --config ~/nats-WAMS/config.conf`

    proc: asyncio.subprocess.Process = await asyncio.create_subprocess_shell(
        'nats-server --config ~/nats-WAMS/config.conf',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    await asyncio.sleep(.5)
    return proc


async def test_setup() -> tuple[Process, nats.NATS, JetStreamContext]:
    proc = await run_the_server()

    nc = await connect_to_nats("localhost:4222", "szakal")
    js, stream = await create_stream(nc, False, STREAM_NAME)

    await clear_subject(js, STREAM_NAME, "stan.broker.alive")

    return proc, nc, js


async def test_broker():
    server, nc, js = await test_setup()

    broker = await MessageBroker.Create("localhost:4222", "szakal")

    broker_task = asyncio.create_task(broker.the_loop())

    await asyncio.sleep(1)

    # Test for the keep-alive message
    last_message = await js.get_last_msg(STREAM_NAME, "stan.broker.alive")
    assert last_message is not None
    assert last_message.headers["format"] in {"json", "pickle"}
    assert last_message.headers["id"] == broker._broker_id.object_id
    assert last_message.data is not None

    broker_info = BrokerInfo.CreateFromSerialized(last_message.data, BrokerInfo, last_message.headers["format"])
    assert broker_info.object_id == broker._broker_id.object_id
    print(broker_info.pretty_print())

    try:
        broker2 = await MessageBroker.Create("localhost:4222", "szakal")
        await broker2.check_for_network_duplicates()

    except NetworkDuplicateError as e:
        assert e.other_object_id == broker._broker_id.object_id

    await broker.shutdown()
    await asyncio.wait_for(broker_task, 0.1)  # Will raise an exception if broker did not shut down in 0.1 second

    try:
        last_message = await js.get_last_msg(STREAM_NAME, "stan.broker.alive")
    except nats.js.errors.NotFoundError:
        last_message = None

    assert last_message is None

    await teardown(server, nc)


async def teardown(proc: Process, nc: nats.NATS):
    await nc.close()
    try:
        proc.terminate()
    except ProcessLookupError:
        pass



if __name__ == '__main__':
    asyncio.run(test_broker(), debug=True)
