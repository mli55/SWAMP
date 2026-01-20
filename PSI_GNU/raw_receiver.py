import time
import socket
import numpy as np
from gnuradio import gr, uhd
from gnuradio import blocks

class CFRReceiver(gr.top_block):
    def __init__(
        self,
        start_freq,
        end_freq,
        step_freq,
        samp_rate,
        rx_gain,
        capture_time=1,                 # Capture for 1 second (默认参数)
        output_file="/home/amelia/cfr_data.bin",
        receiver_addr="0.0.0.0",        # Listen on all interfaces
        receiver_port=9999
    ):
        gr.top_block.__init__(self)

        # Configure the USRP source
        self.usrp_source = uhd.usrp_source(
            ",".join(("type=x300", "")),
            uhd.stream_args(cpu_format="fc32", channels=[0])
        )
        self.usrp_source.set_samp_rate(samp_rate)
        self.usrp_source.set_gain(rx_gain)
        self.usrp_source.set_clock_source("internal")

        # Frequency sweep parameters
        self.start_freq   = start_freq
        self.end_freq     = end_freq
        self.step_freq    = step_freq
        self.current_freq = start_freq
        self.capture_time = capture_time
        self.output_file  = output_file

        # Set initial frequency
        self.usrp_source.set_center_freq(self.start_freq)

        # Vector sink for capturing samples
        self.vector_sink = blocks.vector_sink_c()
        self.connect(self.usrp_source, self.vector_sink)

        # Create a TCP server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Allow immediate reuse of addr/port
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.server_socket.bind((receiver_addr, receiver_port))
        self.server_socket.listen(1)
        print(f"[Receiver] Listening on {receiver_addr}:{receiver_port}")

    def start_measurement(self):
        """
        Main measurement loop:
          1) Start the flow graph (USRP receiving).
          2) Repeatedly accept connections from any transmitter.
          3) For each connection, sweep from start_freq to end_freq:
             - Wait for the transmitter's "start" message
             - Wait 0.2 s (settle time)
             - Capture for `capture_time` seconds
             - Write data
             - Move to next frequency
          4) After each sweep, return to start_freq
          5) Close connection, wait for next connection
          6) To stop, press Ctrl-C or otherwise kill the program.
        """
        self.start()  # Start the receiving flow graph
        print("[Receiver] Flow graph started. Waiting for connections...")

        try:
            while True:
                # Accept one connection from the transmitter
                print("[Receiver] Waiting for a transmitter to connect...")
                conn, addr = self.server_socket.accept()
                print(f"[Receiver] Transmitter connected from {addr}")

                try:
                    # For a new sweep, reset frequency to the start
                    self.current_freq = self.start_freq
                    self.usrp_source.set_center_freq(self.current_freq)

                    # Open the file in append mode so multiple sweeps are stored
                    with open(self.output_file, "ab") as f:
                        while self.current_freq <= self.end_freq:
                            # 1) Wait for a message from the transmitter
                            data = conn.recv(1024)
                            if not data:
                                print("[Receiver] No more data from transmitter. Ending this sweep.")
                                break

                            message = data.decode("utf-8").strip()
                            print(f"[Receiver] Received message: {message}")

                            # 2) Wait 0.2 seconds (settle time)
                            time.sleep(0.2)

                            # 3) Flush old samples from the sink
                            self.vector_sink.reset()

                            # 4) Capture for self.capture_time seconds
                            time.sleep(self.capture_time)

                            # 5) Retrieve samples
                            samples = np.array(self.vector_sink.data(), dtype=np.complex64)
                            self.vector_sink.reset()

                            # 6) Write a frequency marker (real = freq, imag = 0)
                            freq_marker = np.array([np.complex64(self.current_freq + 0j)], dtype=np.complex64)
                            freq_marker.tofile(f)

                            # 7) Write captured samples to file
                            samples.tofile(f)

                            # 8) Increment frequency
                            self.current_freq += self.step_freq

                            # If there's still another frequency to go, set it
                            if self.current_freq <= self.end_freq:
                                self.usrp_source.set_center_freq(self.current_freq)
                            else:
                                print("[Receiver] Frequency sweep complete.")
                                break

                    # After finishing or breaking from the sweep, go back to start_freq
                    self.usrp_source.set_center_freq(self.start_freq)
                    print(f"[Receiver] Returned to origin frequency: {self.start_freq/1e6:.3f} MHz")
                
                finally:
                    # Close the current connection gracefully
                    try:
                        conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    conn.close()
                    print("[Receiver] Connection closed. Ready for next transmitter.\n")

        except KeyboardInterrupt:
            print("\n[Receiver] Caught Ctrl-C. Exiting...")

        finally:
            # Cleanup the flowgraph and socket when exiting the loop
            self.stop()
            self.wait()
            self.server_socket.close()
            print(f"[Receiver] Data saved in {self.output_file}")
            print("[Receiver] Flow graph stopped. Goodbye!")


if __name__ == "__main__":
    start_freq = 2.0e9
    end_freq   = 5.0e9
<<<<<<< HEAD
    step_freq  = 40e6
=======
    step_freq  = 40e6          # 同样改为 40 MHz 步进
>>>>>>> c98dfe04c0fb4eb6ac81766a2426c99d7e7d3f93
    samp_rate  = 500e3
    rx_gain    = 60
    capture_time = 0.5         # 每个频点接收 0.5 秒
    output_file = "/home/amelia/cfr_data.bin"

    receiver = CFRReceiver(
        start_freq, end_freq, step_freq,
        samp_rate, rx_gain, capture_time,
        output_file,
        receiver_addr="0.0.0.0",
        receiver_port=9999
    )

    print("[Receiver] Starting reception flow graph...")
    receiver.start_measurement()
    print("[Receiver] Reception complete.")
