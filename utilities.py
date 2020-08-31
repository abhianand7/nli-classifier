import concurrent
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import types
from tqdm import tqdm


def parallelize_task_by_thread(func, iterable, workers=4):
    # print(next(iterable))
    with ThreadPoolExecutor(max_workers=workers) as executor:

        future_to_url = {executor.submit(func, i): i for i in iterable}
        result = []

        for future in concurrent.futures.as_completed(future_to_url, timeout=90):
            result.append(future.result())
            # print(future.result())
        return result


# spawn of processes are resource intensive, so always try to launch
# the number of workers as many as cores of the system
def parallelize_task_by_process(func, iterable, workers):
    if isinstance(iterable, types.GeneratorType):
        iterable = list(iterable)
        chunksize = len(iterable) // workers
    else:
        chunksize = len(iterable) // workers
    if chunksize < 1:
        chunksize = 1
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # future_to_url = {executor.submit(func, i): i for i in iterable}
        future_to_url = executor.map(func, iterable, chunksize=chunksize)
        result = []

        for future in tqdm(future_to_url, desc='Progress...'):
            if not isinstance(future, type(None)):

                result.append(future)
            # result.append(future.result())
        return result