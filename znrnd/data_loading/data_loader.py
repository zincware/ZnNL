"""
ZnRND: A zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
Module for the data loader.
"""
from znrnd.data_sets.data_set import DataSet
import numpy as np
from typing import List, Tuple
import multiprocessing
from multiprocessing import Queue
import queue
from itertools import cycle


class DataLoader:
    """
    Data loader class.

    The data loader takes a data-set at instantiation and loads pieces of it into
    memory as required. It also allows for preloading of data to reduce latency during
    reading.
    """
    def __init__(
            self,
            dataset: DataSet,
            batch_size: int = 32,
            n_workers: int = 1,
            n_prefetch_batches: int = 1
    ):
        """
        Constructor for the data loader.

        Parameters
        ----------
        dataset : DataSet
                Dataset from which to gather data.
        batch_size : int
                Batch size used in loading.
        n_workers : int
                Number of workers to use in preloading.
        n_prefetch_batches : int
                Number of batches to preload into memory.
        """
        # Set by user
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.n_prefetch_batches = n_prefetch_batches
        self.dataset = dataset

        # Class only
        self._index = 0
        self._output_queue = Queue()
        self._index_queues = []
        self._workers = []
        self._worker_cycle = cycle(range(n_workers))
        self._cache = {}
        self._prefetch_index = 0

        # Start the workers and get some data.
        self._initialize_workers()
        self.prefetch()

    def _initialize_workers(self):
        """
        Initialize the MP workers.

        We need to create a set of workers with the datasets functions so that they can
        be called to grab more data when needed.
        """
        for _ in range(self.n_workers):
            index_queue = Queue()
            worker = multiprocessing.Process(
                target=self._worker_fn,
                args=(self.dataset, index_queue, self._output_queue)
            )
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
            self._index_queues.append(index_queue)

    @staticmethod
    def _collate_fn(batch: List[Tuple]) -> dict:
        """
        Function to collect the data into the correct format.

        Parameters
        ----------
        batch : list
                Batch of data to organize.

        Returns
        -------
        dataset : dict
                Dataset object with {"inputs": np.ndarray, "targets": np.ndarray}
                dict structure.
        """
        return {
            "inputs": np.array([item[0] for item in batch]),
            "targets": np.array([item[1] for item in batch])
        }

    @staticmethod
    def _worker_fn(dataset: DataSet, index_queue: Queue, output_queue: Queue):
        """
        Function for multiprocess workers.

        Returns
        -------
        Updates a queue of data with a new point.
        """
        while True:
            try:
                index = index_queue.get(timeout=0)
            except queue.Empty:
                continue
            if index is None:
                break
            output_queue.put((index, dataset[index]))

    def prefetch(self):
        """
        Prefetch the next N batches.

        Returns
        -------
        Updates the prefetch index triggering the workers to gather more data.
        """
        while (
            self._prefetch_index < len(self.dataset) and self._prefetch_index
            < self._index + self.n_prefetch_batches * self.n_workers * self.batch_size
        ):
            self._index_queues[next(self._worker_cycle)].put(self._prefetch_index)
            self._prefetch_index += 1

    def get(self) -> Tuple:
        """
        Collect data from the dataset.

        Returns
        -------
        item : Tuple
                Batch of data.
        """
        self.prefetch()

        if self._index in self._cache:
            item = self._cache[self._index]
            del self._cache[self._index]
        else:
            while True:
                try:
                    (index, data) = self._output_queue.get(timeout=0)
                except queue.Empty:
                    continue
                if index == self._index:
                    item = data
                else:
                    self._cache[index] = data

        self._index += 1
        return item

    def __next__(self) -> dict:
        """
        Implementation of the next iteration method for the class.

        Overwriting this method makes the class an iterable that we can use in a loop.
        In this case, it will simply load the next batch of data until there is no more
        data to load.

        Returns
        -------
        batch_data : dict
            Dataset object with {"inputs": np.ndarray, "targets": np.ndarray}
                    dict structure.
        """
        # Stop iterating if the index is over the end.
        if self._index >= len(self.dataset):
            raise StopIteration

        # Get the correct batch size
        batch_size = min(len(self.dataset) - self._index, self.batch_size)

        return self._collate_fn([self.get() for _ in range(batch_size)])

    def __iter__(self):
        """
        Iterator function for looping through the dataset.

        Returns
        -------
        Reset the data loader after the data is finished.
        """
        self._index = 0
        self._cache = {}
        self._prefetch_index = 0
        self.prefetch()

        return self

    def __del__(self):
        """
        Override the delete method.

        When a class is no longer used and Python runs garbage collection, we need to be
        sure that all the workers have stopped correctly. Therefore, the del method
        is overwritten.

        Returns
        -------
        """
        try:
            for i, w in enumerate(self._workers):
                self._index_queues[i].put(None)
                w.join(timeout=5.0)
            for q in self._index_queues:
                q.cancel_join_thread()
                q.close()
            self._output_queue.cancel_join_thread()
            self._output_queue.close()
        finally:
            for w in self._workers:
                if w.is_alive():
                    w.terminate()
