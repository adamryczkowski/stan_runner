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

    await asyncio.sleep(2)

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
    await asyncio.wait_for(broker_task, 1)  # Will raise an exception if broker did not shut down in 0.1 second

    try:
        last_message = await js.get_last_msg(STREAM_NAME, "stan.broker.alive")
    except nats.js.errors.NotFoundError:
        last_message = None

    assert last_message is not None
    assert last_message.headers["format"] in {"json", "pickle"}
    assert last_message.headers["state"] == "shutdown"


    await teardown(server, nc)


async def test_worker():
    server, nc, js = await test_setup()

    broker = await MessageBroker.Create("localhost:4222", "szakal")
    broker_task = asyncio.create_task(broker.the_loop())

    print("Starting worker...")
    worker = await NatsWorker.Create("localhost:4222", "szakal")
    print("Worker created, entering the loop...")
    worker_task = asyncio.create_task(worker.the_loop())

    print("Waiting for broker to notice the worker...")
    await asyncio.sleep(0.5)

    print(" Check if worker produces advertisements (alive messages)...")
    last_message = await js.get_last_msg(STREAM_NAME, f"stan.worker_advert.{worker.worker_id}")
    assert last_message is not None
    assert last_message.headers["format"] in {"json", "pickle"}
    assert last_message.headers["id"] == worker.worker_id
    assert last_message.data is not None
    print("...pass\n")

    print("Check if the advertisement object is correct...")
    worker_info = WorkerInfo.CreateFromSerialized(last_message.data, WorkerInfo, last_message.headers["format"])
    assert isinstance(worker_info, WorkerInfo)
    assert worker_info.object_id == worker.worker_id
    print("...pass\n")

    print("Check if broker notices the worker, and if it is the correct one...")
    assert len(broker.workers) == 1
    broker_worker_id = next(iter(broker.workers.keys()))
    assert worker.worker_id == broker_worker_id
    worker_from_broker = broker.workers[broker_worker_id]
    assert worker_from_broker.object_id == worker.worker_id
    assert worker_from_broker.pretty_print() == worker_info.pretty_print()
    print("...pass\n")

    print("Another worker spawn.")
    worker2 = await NatsWorker.Create("localhost:4222", "szakal")
    worker2_task = asyncio.create_task(worker2.the_loop())
    await asyncio.sleep(0.5) # Make sure broker has time to notice it

    print("Check if broker notices the other worker, and if it is the correct one...")
    assert len(broker.workers) == 2
    assert worker2.worker_id in broker.workers
    worker_from_broker = broker.workers[worker2.worker_id]
    assert worker_from_broker.object_id == worker2.worker_id
    assert worker_from_broker.pretty_print()
    print("...pass\n")




    print("Shutting down worker 1...")
    await worker.shutdown()
    await asyncio.wait_for(worker_task, 0.1)  # Will raise an exception if broker did not shut down in 0.1 second
    await asyncio.sleep(0.5)

    print("Check if broker notices the worker missing...")
    assert len(broker.workers) == 1
    assert worker2.worker_id in broker.workers
    print("...pass\n")


    print("Shutting down worker 2...")
    await worker2.shutdown()
    await asyncio.wait_for(worker2_task, 0.1)  # Will raise an exception if broker did not shut down in 0.1 second

    await asyncio.sleep(0.5)
    assert len(broker.workers) == 0


    print("Shutting down the broker...")
    await broker.shutdown()
    await asyncio.wait_for(broker_task, 0.1)  # Will raise an exception if broker did not shut down in 0.1 second

    print("Tearing down the server...")
    await teardown(server, nc)


async def kill_nets_server(proc: Process):
    try:
        # proc.terminate()
        print("Trying to terminate the NATS server process...")
        try:
            parent = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            print("No NATS servers managed by us. Exiting.")
            return

        for child in parent.children(recursive=True):
            print("Sending sigint to child process")
            child.send_signal(2)  # sigint
            child.wait(1)
        if parent.is_running():
            await proc.wait()
            if parent.is_running():
                print("Stubborn process. Killing it.")
                parent.terminate()
        await proc.wait()
    except ProcessLookupError:
        pass


async def teardown(proc: Process, nc: nats.NATS):

    # Close all asyncio tasks
    for task in asyncio.all_tasks():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


    await nc.close()
    await asyncio.sleep(0)

    await kill_nets_server(proc)





if __name__ == '__main__':
    # asyncio.run(test_broker(), debug=True)
    asyncio.run(test_worker(), debug=True)
    # asyncio.get_event_loop().close()
