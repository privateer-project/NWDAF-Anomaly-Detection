import pynq
import numpy as np
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Dict, Type, List, Optional, Tuple
import time
import inspect
import threading
import queue

from .alveo_power_scraper import ALVEOPowerScraper


@dataclass
class AlveoRunnerParameters:
    """
    Dataclass to hold parameters for configuring the ML Model.
    """

    input_buffer_elements: int
    output_buffer_elements: int
    input_t: type = float  # Data type for the input.
    result_t: type = float  # Data type for the output.
    kernel_name: str = None


class AlveoRunner:
    """
    Class to interface with an ML model implemented on an Alveo FPGA device using PYNQ.
    """

    def __init__(
        self,
        bitstream_path: str,
        parameters: AlveoRunnerParameters,
        device: pynq.Device = None,
        device_bus: Optional[str] = None,
    ) -> None:
        """
        Initializes the Autoencoder model on the FPGA by loading the specified bitstream.

        Args:
            bitstream_path (str): Path to the bitstream file to load onto the FPGA.
            parameters (AlveoRunnerParameters): Parameters for configuring the ML model.
            device (pynq.Device, optional): Specific FPGA device to use. If None, the default device is used.
        """
        # Initialize timing dictionary to track performance metrics
        self.timings = {  # Timing data in nanoseconds
            "initialize": None,
            "allocate_buffers": None,
            "runs": [],
        }

        # Measure time taken to initialize the FPGA overlay
        initialize_start = time.perf_counter()
        # Load the overlay (bitstream) onto the FPGA
        if device is None:
            self.overlay = pynq.Overlay(bitstream_path)
        else:
            self.overlay = pynq.Overlay(bitstream_path, device=device)
        initialize_end = time.perf_counter()
        self.timings["initialize"] = initialize_end - initialize_start
        print("Initialized")

        # self.kernel = self.overlay.lstm_1  # Reference to the Autoencoder kernel on the FPGA
        # Store parameters and set up the kernel and buffers
        self.parameters = parameters
        if self.parameters.input_buffer_elements <= 0:
            raise ValueError(
                "AlveoRunnerParameters.input_buffer_elements must be positive."
            )
        if self.parameters.output_buffer_elements < 0:
            raise ValueError(
                "AlveoRunnerParameters.output_buffer_elements cannot be negative."
            )
        if self.parameters.kernel_name is None:
            # Get default kernel IP (works when only one kernel exists)
            try:
                self.kernel = getattr(self.overlay, next(iter(self.overlay.ip_dict)))
            except Exception as e:
                raise AssertionError(
                    f"Hmmm  self.kernel = getattr(self.overlay, next(iter(self.overlay.ip_dict))) failed,here is the error {e}\n here is the help {self.help()}\n and the ip_dict {self.get_ip_dict()}"
                )
        else:
            self.kernel = getattr(self.overlay, self.parameters.kernel_name)

        self.pprint_ip_dict()  # Print information about available IP cores
        self.print_kernel_signature()  # Print the kernel's function signature

        self.input_buffer = None
        self.output_buffer = None

        # Allocate input and output buffers on the FPGA
        allocate_buffers_start = time.perf_counter()
        self._allocate_buffers()
        allocate_buffers_end = time.perf_counter()
        self.timings["allocate_buffers"] = allocate_buffers_end - allocate_buffers_start
        print("Allocated Buffers")

        # Initialize power_scraper
        self.power_scraper: Optional[ALVEOPowerScraper] = None
        self.device_bus = device_bus

        if self.device_bus:
            print(
                f"Attempting to initialize ALVEOPowerScraper for device_bus: {self.device_bus}..."
            )
            try:
                # Default XRT version "2.13" is used by ALVEOPowerScraper if not specified
                self.power_scraper = ALVEOPowerScraper(device_bus=self.device_bus)
            except ValueError as e:
                print(f"Warning: Failed to initialize ALVEOPowerScraper. {e}")
                print("Power scraping will be disabled.")
                self.power_scraper = None
            except RuntimeError as e:
                print(f"Warning: ALVEOPowerScraper encountered a runtime error. {e}")
                print("Power scraping will be disabled.")
                self.power_scraper = None
            except Exception as e:
                print(
                    f"Warning: An unexpected error occurred while initializing ALVEOPowerScraper: {e}"
                )
                print("Power scraping will be disabled.")
                self.power_scraper = None
        else:
            print(
                "No device_bus specified for AlveoRunner. Power scraping is disabled."
            )

        print("Succesfully loaded ALVEO model!")

    def _allocate_buffers(self) -> None:
        """
        Allocates memory buffers for input and output on the FPGA.
        """
        # Allocate buffers with specified data types
        self.input_buffer = pynq.allocate(
            shape=(self.parameters.input_buffer_elements,),
            dtype=convert_types(self.parameters.input_t),
        )
        self.output_buffer = pynq.allocate(
            shape=(self.parameters.output_buffer_elements),
            dtype=convert_types(self.parameters.result_t),
        )

    def run(self, input_darray: np.ndarray) -> np.ndarray:
        """
        Runs the Autoencoder model on the FPGA using the provided input data.

        Args:
            input_darray (np.ndarray): Input data array.

        Returns:
            np.ndarray: Output data from the Autoencoder model.
        """
        # Copy input data to the FPGA buffer and sync it
        self.input_buffer[:] = input_darray
        self.input_buffer.sync_to_device()
        # Execute the Autoencoder kernel on the FPGA
        self.kernel.call(self.input_buffer, self.output_buffer)
        # Sync the output buffer from the FPGA
        self.output_buffer.sync_from_device()
        return self.output_buffer.copy()

    def timed_run(self, input_darray: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Runs the Autoencoder model on the FPGA with timing measurement and optional verbosity.

        Args:
            input_darray (np.ndarray): Input data array that will be passed to the Autoencoder model.
                Must have the same shape and data type as the model's input buffer.
            verbose (bool, optional): If True, prints the runtime of the model execution in milliseconds.
                Defaults to True.

        Returns:
            np.ndarray: Output data from the Autoencoder model as retrieved from the output buffer on the FPGA.

        Raises:
            AssertionError: If the shape or dtype of `input_darray` does not match the expected input buffer specifications.
        """
        # Check that the input matches the expected shape and type
        assert self.input_buffer.shape == input_darray.shape, (
            f"Input shape mismatch expected: {self.input_buffer.shape} got: {input_darray.shape}"
        )
        assert self.input_buffer.dtype == input_darray.dtype, (
            f"Input type mismatch expected: {self.input_buffer.dtype} got: {input_darray.dtype}"
        )

        # Measure the runtime of the model execution
        run_start = time.perf_counter()
        output_buffer = self.run(input_darray)
        run_end = time.perf_counter()

        # Store the timing of the run
        self.timings["runs"].append(run_end - run_start)
        # Print runtime if verbose is enabled
        if verbose:
            print(f"Runtime: {self.timings['runs'][-1] * 1000:.2f} ms")

        return output_buffer

    def run_vector(
        self,
        input_vector: np.ndarray,
        output_shape: Optional[Tuple[int, ...]] = None,
        timed: bool = False,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Processes an input vector by flattening it, dividing it into segments
        compatible with the model's input buffer size, running the model on each
        segment, and then optionally reshaping the concatenated results.

        Args:
            input_vector (np.ndarray): The input data array. Can be multi-dimensional.
            output_shape (Optional[Tuple[int, ...]], optional):
                If provided, the flat output vector will be reshaped to this shape.
                The total number of elements in output_shape must match the total
                number of elements in the flat output.
                If None (default), the flat output vector is returned.
            timed (bool, optional): If True, measures and prints execution time for
                                     each segment. Defaults to False.
            verbose (bool, optional): If True and `timed` is True, prints runtime.
                                      Defaults to False.

        Returns:
            np.ndarray: The processed output data. If output_shape is provided and
                        valid, it's reshaped; otherwise, it's the flat output vector.

        Raises:
            ValueError: If input_buffer_elements is not positive, or if output_shape
                        is provided but its product doesn't match the flat output size.
            AssertionError: If the total number of elements in the flattened input
                            is not a multiple of self.parameters.input_buffer_elements.
        """

        # 1) Reshape the input and flattening it (run inputs must be flattened)
        input_flat = input_vector.flatten()

        # 2) Check if the len of input vector is a multiple of input_buffer_elements.
        if len(input_flat) % self.parameters.input_buffer_elements != 0:
            raise AssertionError(
                f"Flattened input vector length ({len(input_flat)}) must be a multiple of "
                f"input_buffer_elements ({self.parameters.input_buffer_elements})."
            )

        num_segments = len(input_flat) // self.parameters.input_buffer_elements

        # 3) Create the flattened output_vector
        output_flat_len = num_segments * self.parameters.output_buffer_elements
        output_vector_flat = np.empty(
            shape=(output_flat_len,), dtype=convert_types(self.parameters.result_t)
        )

        # 4) Do the running of each individual vector using the run or timed_run function
        for i in range(num_segments):
            input_segment_start = i * self.parameters.input_buffer_elements
            input_segment_end = (
                input_segment_start + self.parameters.input_buffer_elements
            )
            current_input_segment = input_flat[input_segment_start:input_segment_end]

            # The run/timed_run methods expect input of shape (self.parameters.input_buffer_elements,)
            # and self.input_buffer is already allocated with this shape.

            if timed:
                segment_result = self.timed_run(current_input_segment, verbose)
            else:
                segment_result = self.run(current_input_segment)

            output_segment_start = i * self.parameters.output_buffer_elements
            output_segment_end = (
                output_segment_start + self.parameters.output_buffer_elements
            )
            output_vector_flat[output_segment_start:output_segment_end] = segment_result

        # 5) Reshape the output_vector if output_shape is provided
        if output_shape is not None:
            if not isinstance(output_shape, tuple):
                raise TypeError(
                    f"output_shape must be a tuple, but got {type(output_shape)}."
                )

            expected_total_elements = np.prod(output_shape)
            if expected_total_elements != output_vector_flat.size:
                raise ValueError(
                    f"The total number of elements in the provided output_shape {output_shape} "
                    f"(product: {expected_total_elements}) does not match the size of the "
                    f"flat output vector ({output_vector_flat.size})."
                )
            try:
                return output_vector_flat.reshape(output_shape)
            except ValueError as e:  # Catch potential reshape errors from numpy
                raise ValueError(
                    f"Failed to reshape flat output to {output_shape}. Numpy error: {e}"
                )
        else:
            # Return the flat output by default
            return output_vector_flat

    def print_timings(self, verbose=False) -> None:
        """
        Prints the recorded performance timings for various stages of the autoencoder operation, including initialization,
        buffer allocation, and individual runs.

        Args:
            verbose (bool, optional): If True, prints detailed timing information for each individual run. Defaults to False.
        """
        print(f"Initialize time: {self.timings['initialize'] * 1000:.2f} ms")
        print(
            f"Allocate buffers time: {self.timings['allocate_buffers'] * 1000:.2f} ms"
        )
        if len(self.timings["runs"]) != 0:  # Check if there are any recorded runs
            print(
                f"Average run: {sum(self.timings['runs']) / len(self.timings['runs']) * 1000:.2f} ms"
            )
            print(f"Number of runs: {len(self.timings['runs'])}")
            if verbose:
                for i, run in enumerate(self.timings["runs"]):
                    print(f"Run {i}: {run * 1000:.2f} ms")

    def help(self) -> None:
        """
        Prints information about the Autoencoder overlay, including its type, docstring, attributes, and methods.
        """
        # Get and print the type of the overlay object
        print(f"Type: {type(self.overlay)}\n")

        # Retrieve and print the docstring of the overlay object, if available
        docstring = inspect.getdoc(self.overlay)
        if docstring:
            print(f"Docstring:\n{docstring}\n")
        else:
            print("Docstring: None\n")

        # List and print all attributes and methods of the overlay
        attributes = dir(self.overlay)
        print(f"Attributes and Methods:\n{attributes}\n")

    def get_ip_dict(self) -> Dict[str, Any]:
        """
        Retrieves the dictionary of IP cores available in the overlay.

        Returns:
            Dict[str, Any]: Dictionary containing IP cores information.
        """
        return self.overlay.ip_dict

    def get_mem_dict(self) -> Dict[str, Any]:
        """
        Retrieves the dictionary of memory blocks available in the overlay.

        Returns:
            Dict[str, Any]: Dictionary containing memory blocks information.
        """
        return self.overlay.mem_dict

    def print_kernel_signature(self) -> None:
        """
        Prints the function signature of the Autoencoder kernel.
        """
        print(self.kernel.signature)

    def pprint_ip_dict(self) -> None:
        """
        Pretty prints the dictionary of IP cores available in the overlay.
        """
        pprint(self.overlay.ip_dict, indent=4)

    def print_used_mem_dict(self) -> None:
        """
        Prints information about the used memory blocks in the overlay.
        """
        for key in self.overlay.mem_dict.keys():
            if self.overlay.mem_dict[key]["used"] == 1:
                print(f"Key: {key}, {self.overlay.mem_dict[key]}")

    def clean_class(self) -> None:
        """
        Cleans up resources by deleting buffers and freeing the FPGA overlay.
        """
        del self.input_buffer  # Deletes the input buffer from memory
        del self.output_buffer  # Deletes the output buffer from memory
        self.overlay.free()  # Frees the overlay resources

    def get_average_time_alveo(
        self, input_vector: np.ndarray, iterations: int = 1000, warmup: int = 10
    ) -> float:
        total_time = 0
        # Copy input data to the FPGA buffer and sync it
        self.input_buffer[:] = input_vector
        self.input_buffer.sync_to_device()
        for _ in range(warmup):
            _ = self.kernel.call(self.input_buffer, self.output_buffer)
        # self.output_buffer.sync_from_device()
        for i in range(iterations):
            start = time.perf_counter()
            _ = self.kernel.call(self.input_buffer, self.output_buffer)
            # self.output_buffer.sync_from_device()
            end = time.perf_counter()
            total_time += end - start
        average_time_ms = total_time / iterations * 1000  # (ms)
        return average_time_ms

    def get_average_time_alveo_transfers(
        self, input_vector: np.ndarray, iterations: int = 1000, warmup: int = 10
    ) -> float:
        total_time = 0
        # Copy input data to the FPGA buffer and sync it
        self.input_buffer[:] = input_vector
        self.input_buffer.sync_to_device()
        for _ in range(warmup):
            _ = self.kernel.call(self.input_buffer, self.output_buffer)
        self.output_buffer.sync_from_device()
        for i in range(iterations):
            start = time.perf_counter()
            # _ = self.run(input_vector) This includes host -> buffer copies, slow
            self.input_buffer.sync_to_device()
            _ = self.kernel.call(self.input_buffer, self.output_buffer)
            self.output_buffer.sync_from_device()
            end = time.perf_counter()
            total_time += end - start
        average_time_ms = total_time / iterations * 1000  # (ms)
        return average_time_ms

    def _collect_power_data_thread(
        self,
        event: threading.Event,
        timeout_seconds: float,
        power_data_queue: queue.Queue,
    ) -> None:
        """
        Collects power data from the FPGA in a separate thread for a specified duration.

        Args:
            event (threading.Event): Event to signal the end of data collection.
            timeout_seconds (float): Duration in seconds for which to collect power data.
            power_data_queue (queue.Queue): Queue to store the collected power data.
        """
        time.sleep(1)  # To get correct power data
        power_data_list = []
        start = time.perf_counter()
        end = start
        while end - start < timeout_seconds:
            try:
                power_data_list.append(self.power_scraper.get_power())
            except Exception as e:
                print(f"self.power_scraper.get_power() got this error: {e}")
            end = time.perf_counter()
        event.set()
        power_data_queue.put(power_data_list)
        return

    def _continuous_runs_thread(self, event: threading.Event) -> None:
        """
        Continuously runs the Autoencoder kernel on the FPGA until the event is set.

        Args:
            event (threading.Event): Event to signal when to stop the continuous runs.
        """
        self.input_buffer[:] = np.random.random(
            (self.parameters.input_buffer_elements,)
        ).astype(convert_types(self.parameters.input_t))
        self.input_buffer.sync_to_device()
        while not event.is_set():
            self.kernel.call(self.input_buffer, self.output_buffer)
        self.output_buffer.sync_from_device()
        return

    def _continuous_runs_transfers_thread(self, event: threading.Event) -> None:
        """
        Continuously runs the Autoencoder kernel on the FPGA until the event is set.

        Args:
            event (threading.Event): Event to signal when to stop the continuous runs.
        """
        self.input_buffer[:] = np.random.random(
            (self.parameters.input_buffer_elements,)
        ).astype(convert_types(self.parameters.input_t))
        while not event.is_set():
            self.input_buffer.sync_to_device()
            self.kernel.call(self.input_buffer, self.output_buffer)
            self.output_buffer.sync_from_device()
        return

    def get_power_data(
        self, seconds: float = 10.0, transfers: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Collects power data from the FPGA during continuous runs for a specified duration.

        Args:
            seconds (float, optional): Duration in seconds to collect power data. Defaults to 10.0 seconds.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing power data collected during the run.
        """
        if self.power_scraper is None:
            print(
                "Power scraping is disabled (AlveoRunner.power_scraper is None). Cannot collect power data."
            )
            return []
        event = threading.Event()  # Event for signaling
        power_data_queue = queue.Queue()  # Queue for communication
        continuous_power_data = threading.Thread(
            target=self._collect_power_data_thread,
            args=(event, seconds, power_data_queue),
        )
        if transfers:
            continuous_runs = threading.Thread(
                target=self._continuous_runs_transfers_thread, args=(event,)
            )
        else:
            continuous_runs = threading.Thread(
                target=self._continuous_runs_thread, args=(event,)
            )
        continuous_runs.start()
        continuous_power_data.start()
        continuous_runs.join()
        power_data = power_data_queue.get()
        continuous_power_data.join()

        return power_data


def convert_types(c_type: Type) -> Type[np.generic]:
    """
    Converts a Python type to the corresponding NumPy type used for buffer allocation on the FPGA.

    Args:
        c_type (Type): The Python data type to convert (e.g., int, float).

    Returns:
        Type[np.generic]: The corresponding NumPy data type (e.g., np.int32, np.float32).

    Raises:
        ValueError: If the provided type is not supported for conversion.
    """
    if c_type is int:
        return np.int32
    if c_type is float:
        return np.float32
    raise TypeError(
        f"Unsupported type: {c_type}. Cannot be converted to a NumPy dtype by placeholder"
    )