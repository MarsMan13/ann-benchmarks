from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Optional
import docker
import psutil
import re

import numpy

class BaseANN(object):
    """Base class/interface for Approximate Nearest Neighbors (ANN) algorithms used in benchmarking."""

    def done(self) -> None:
        """Clean up BaseANN once it is finished being used."""
        pass

    def get_memory_usage(self) -> Optional[float]:
        """Returns the current memory usage of this ANN algorithm instance in kilobytes.

        Returns:
            float: The current memory usage in kilobytes (for backwards compatibility), or None if
                this information is not available.
        """

        return psutil.Process().memory_info().rss / 1024
    
    def get_memory_usage_by_program(self, target_name: str) -> Optional[float]:
        total_memory = 0
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = " ".join(process.info['cmdline']) if process.info['cmdline'] else ""
                if target_name in cmdline:
                    memory_kb = process.memory_info().rss / 1024
                    total_memory += memory_kb
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_memory
    
    def get_memory_usage_of_docker_container(self, container_name: str) -> Optional[float]:
        client = docker.from_env()
        try:
            container = client.containers.get(container_name)
            memory_stats = container.stats(stream=False)['memory_stats']
            return memory_stats['usage'] / 1024
        except docker.errors.NotFound:
            print(f"Container {container_name} not found.")
            return None
        except Exception as e:
            print(f"Error while getting container {container_name}: {e}")
    
    def get_memory_usage_by_program_from_docker_container(self, container_name:str, target_name: str) -> Optional[float]:
        client = docker.from_env()
        try:
            container = client.containers.get(container_name)
            
            exec_command = f"/bin/bash -c 'ps aux | grep {target_name} | grep -v grep'"
            print(exec_command)
            result = container.exec_run(exec_command)
            output = result.output.decode("utf-8")
            
            if not output.strip():
                print(f"Process {target_name} not found in container {container_name}")
                return None
            print(f"Process {target_name} found in container {container_name}!")
            total_memory = 0
            for line in output.splitlines():
                match = re.search(r"\s+(\d+)\s+", line)
                if match:
                    pid = match.group(1)
                    mem_command = f"cat /proc/{pid}/statm"
                    mem_result = container.exec_run(mem_command)
                    mem_output = mem_result.output.decode("utf-8")
                    if mem_output.strip():
                        print(f"Memory output: {mem_output.split()}")
                        memory_kb = int(mem_output.split()[1]) * 4
                        total_memory += memory_kb
                        print(f"\n\nmemory_kb: {memory_kb}")
                        print(f"total_memory: {total_memory}\n\n")
            return total_memory
        except docker.errors.NotFound:
            print(f"Container {container_name} not found.")
            return None
        except Exception as e:
            print(f"Error while getting container {container_name}: {e}")
            return None

    def fit(self, X: numpy.array) -> None:
        """Fits the ANN algorithm to the provided data. 

        Note: This is a placeholder method to be implemented by subclasses.

        Args:
            X (numpy.array): The data to fit the algorithm to.
        """
        pass

    def query(self, q: numpy.array, n: int) -> numpy.array:
        """Performs a query on the algorithm to find the nearest neighbors. 

        Note: This is a placeholder method to be implemented by subclasses.

        Args:
            q (numpy.array): The vector to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return.

        Returns:
            numpy.array: An array of indices representing the nearest neighbors.
        """
        return []  # array of candidate indices

    def batch_query(self, X: numpy.array, n: int) -> None:
        """Performs multiple queries at once and lets the algorithm figure out how to handle it.

        The default implementation uses a ThreadPool to parallelize query processing.

        Args:
            X (numpy.array): An array of vectors to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return for each query.
        Returns: 
            None: self.get_batch_results() is responsible for retrieving batch result
        """
        pool = ThreadPool()
        self.res = pool.map(lambda q: self.query(q, n), X)

    def get_batch_results(self) -> numpy.array:
        """Retrieves the results of a batch query (from .batch_query()).

        Returns:
            numpy.array: An array of nearest neighbor results for each query in the batch.
        """
        return self.res

    def get_additional(self) -> Dict[str, Any]:
        """Returns additional attributes to be stored with the result.

        Returns:
            dict: A dictionary of additional attributes.
        """
        return {}

    def __str__(self) -> str:
        return self.name