# This is a general unit test for the nats system.
#
# It runs a NATS server in a separate thread.
import asyncio
from asyncio.subprocess import Process

from nats.js.errors import NotFoundError
import psutil
from stan_runner import *


async def run_the_server() -> Process:
    # Server is run with a command `nats-server --config ~/nats-WAMS/config.conf`

    # proc: asyncio.subprocess.Process = \
    #     await asyncio.create_subprocess_exec("/usr/local/bin/nats-server",
    #                                          "--config ~/nats-WAMS/config.conf",
    #                                          stdout=asyncio.subprocess.PIPE,
    #                                          stderr=asyncio.subprocess.PIPE)
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

    await asyncio.sleep(0.5)

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


async def test_worker():
    server, nc, js = await test_setup()

    broker = await MessageBroker.Create("localhost:4222", "szakal")
    broker_task = asyncio.create_task(broker.the_loop())

    worker = await NatsWorker.Create("localhost:4222", "szakal")
    worker_task = asyncio.create_task(worker.the_loop())

    await asyncio.sleep(0.5)
    last_message = await js.get_last_msg(STREAM_NAME, f"stan.worker_advert.{worker.worker_id}")
    assert last_message is not None
    assert last_message.headers["format"] in {"json", "pickle"}
    assert last_message.headers["id"] == worker.worker_id
    assert last_message.data is not None

    worker_info = WorkerInfo.CreateFromSerialized(last_message.data, WorkerInfo, last_message.headers["format"])
    assert worker_info.object_id == worker.worker_id

    assert len(broker.workers) == 1
    broker_worker_id = next(iter(broker.workers.keys()))
    assert worker.worker_id == broker_worker_id
    worker_from_broker = broker.workers[broker_worker_id]
    assert worker_from_broker.object_id == worker.worker_id
    assert worker_from_broker.pretty_print() == worker_info.pretty_print()

    await worker.shutdown()
    await asyncio.wait_for(worker_task, 0.1)  # Will raise an exception if broker did not shut down in 0.1 second

    await asyncio.sleep(0.5)
    assert len(broker.workers) == 0


    await broker.shutdown()
    await asyncio.wait_for(broker_task, 0.1)  # Will raise an exception if broker did not shut down in 0.1 second

    await teardown(server, nc)




async def teardown(proc: Process, nc: nats.NATS):

    # Close all asyncio tasks
    for task in asyncio.all_tasks():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    try:
        # proc.terminate()
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            child.send_signal(2)  # sigint
            child.wait(1)
        if parent.is_running():
            await proc.wait()
            if parent.is_running():
                parent.terminate()

        proc.kill()
    except ProcessLookupError:
        pass


    await nc.close()

    await asyncio.sleep(0)



if __name__ == '__main__':
    # asyncio.run(test_broker(), debug=True)
    asyncio.run(test_worker(), debug=True)
    # asyncio.get_event_loop().close()
