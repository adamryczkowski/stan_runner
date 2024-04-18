from nats import NATS
from nats.aio.client import Client as NATSClient
from nats.js import JetStreamContext
from nats.js.api import StreamConfig, StorageType, DiscardPolicy, StreamInfo
from nats.js.errors import NotFoundError
import asyncio
import traceback


def create_stream(nats_connection: str, permanent_storage: bool, stream_name: str,
                  max_bytes=1024 * 1024 * 1024, ) -> StreamInfo:
    nc = NATS()
    js = nc.jetstream()

    config: StreamConfig = StreamConfig(
        name=stream_name,
        storage=StorageType.FILE if permanent_storage else StorageType.MEMORY,
        max_msgs=100000,  # Max 100k unprocessed test cases.
        max_age=7 * 24 * 60 * 60 * 1_000_000_000,  # 7 days
        no_ack=False,  # We need acks for the task stream
        max_consumers=-1,  # Unlimited number of consumers
        max_bytes=max_bytes,
        max_msg_size=max_bytes,
        discard=DiscardPolicy.OLD,
        duplicate_window=0,
        description="Task stream for scenario run. Each message is a distinct scenario to be run.",
        subjects=[f"stan.task.>"]
    )

    async def create_stream_async() -> StreamInfo | None:
        try:
            await nc.connect(f"0.0.0.0:43579", user="local",
                             password="zFoHRl3fLLniRt03ynfCfnkim33oTtcw", max_reconnect_attempts=1)

            try:
                stream = await js.stream_info(name=stream_name)
            except NotFoundError as e:
                print(f"Making a new stream {stream_name}")
                stream = await js.add_stream(name=stream_name,
                                             storage=StorageType.FILE if permanent_storage else StorageType.MEMORY,
                                             max_msgs=100000,  # Max 100k unprocessed test cases.
                                             # max_age=str(int(7 * 24 * 60 * 60 * 1_000_000_000)),  # 7 days
                                             no_ack=False,  # We need acks for the task stream
                                             max_consumers=-1,  # Unlimited number of consumers
                                             max_bytes=max_bytes,
                                             max_msg_size=max_bytes,
                                             discard=DiscardPolicy.OLD,
                                             duplicate_window=0,
                                             description="Task stream for scenario run. Each message is a distinct scenario to be run.",
                                             subjects=[f"stan.task.>"]
                                             )
        except Exception as e:
            # Prints error and the stack trace:
            print(e)
            traceback.print_exc()
            return None
        return stream

    loop = asyncio.get_event_loop()
    stream = loop.run_until_complete(create_stream_async())
    async def close():
        loop.stop()

    loop.run_until_complete(close())


    return stream
