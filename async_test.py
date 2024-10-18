import asyncio
from pylsl import StreamInlet, resolve_stream
import numpy as np

eeg_data = None


async def pull_eeg_data():
    global eeg_data

    streams = resolve_stream('type', 'EEG')

    inlet = StreamInlet(streams[0], max_buflen=1)
    sampling_rate = 256
    buffer_length = 1
    n_channels = 4

    while True:
        chunk, timestamps = inlet.pull_chunk(timeout=0.01, max_samples=sampling_rate * buffer_length)

        if chunk:
            eeg_data = np.array(chunk)

        await asyncio.sleep(0.001)


async def print_eeg_data():
    global eeg_data

    while True:
        if eeg_data is not None:
            break

        await asyncio.sleep(1)


async def main():
    eeg_task = asyncio.create_task(pull_eeg_data())
    print_task = asyncio.create_task(print_eeg_data())

    await asyncio.gather(eeg_task, print_task)


if __name__ == "__main__":
    asyncio.run(main())
