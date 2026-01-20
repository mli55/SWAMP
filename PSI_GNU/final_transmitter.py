import time
import socket
from gnuradio import gr, uhd
from gnuradio import analog

class CFRTransmitter(gr.top_block):
    def __init__(
        self,
        start_freq,
        end_freq,
        step_freq,
        samp_rate,
        tx_gain,
        receiver_ip="192.168.10.2",   # Replace with actual receiver IP
        receiver_port=9999
    ):
        gr.top_block.__init__(self)

        # USRP Sink
        self.usrp_sink = uhd.usrp_sink(
            ",".join(("type=x300", "")),
            uhd.stream_args(cpu_format="fc32", channels=[0])
        )
        self.usrp_sink.set_samp_rate(samp_rate)
        self.usrp_sink.set_gain(tx_gain)
        self.usrp_sink.set_clock_source("internal")

        # Frequency sweep parameters
        self.start_freq = start_freq
        self.end_freq   = end_freq
        self.step_freq  = step_freq
        self.current_freq = start_freq

        # Simple sine wave source (1 kHz tone)
        self.freq_source = analog.sig_source_c(
            samp_rate,
            analog.GR_SIN_WAVE,
            1e3,  # 1 kHz
            1.0,  # amplitude
            0.0
        )
        self.connect(self.freq_source, self.usrp_sink)

        # Networking info
        self.receiver_ip   = receiver_ip
        self.receiver_port = receiver_port
        self.ack_timeout   = 5  # seconds to wait for receiver acknowledgment

    def start_transmission(self):
        """
        1. Start flow graph
        2. Connect to receiver
        3. For each frequency step:
           - Set center freq
           - Send "start" message
           - Wait for receiver "ready" acknowledgment
        4. After sweep, set frequency back to origin
        5. Close socket, stop flow
        """
        print("[Transmitter] Starting transmission...")
        self.start()  # Start the transmitter flowgraph

        # Create TCP client socket to notify the receiver
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.receiver_ip, self.receiver_port))
        s.settimeout(self.ack_timeout)
        print(f"[Transmitter] Connected to receiver at {self.receiver_ip}:{self.receiver_port}")

        freq = self.start_freq
        while freq <= self.end_freq:
            # 1) Set the frequency
            self.usrp_sink.set_center_freq(freq)
            print(f"[Transmitter] Now transmitting at {freq/1e6:.3f} MHz")

            # 2) Notify the receiver to start capturing
            message = f"start_freq_{freq}"
            s.sendall(message.encode("utf-8"))

            # 3) Wait for receiver acknowledgment before moving on
            try:
                ack = s.recv(1024)
                if ack.decode("utf-8").strip() != "ready":
                    print(f"[Transmitter] Unexpected response: {ack}")
            except socket.timeout:
                print("[Transmitter] Timed out waiting for receiver. Aborting sweep.")
                break

            # 4) Move to next frequency
            freq += self.step_freq

        # Return to origin frequency
        self.usrp_sink.set_center_freq(self.start_freq)
        print(f"[Transmitter] Returned to origin frequency: {self.start_freq/1e6:.3f} MHz")

        # Cleanup
        s.close()
        self.stop()
        self.wait()  # Wait for the flowgraph to stop cleanly
        print("[Transmitter] Transmission complete. Waiting for next command...\n")


if __name__ == "__main__":
    # Adjustable parameters
    start_freq = 2.0e9
    end_freq   = 5.0e9
    step_freq  = 40e6     # 40 MHz step
    samp_rate  = 500e3
    tx_gain    = 60

    # Instantiate once (reuse the top_block).
    transmitter = CFRTransmitter(
        start_freq, end_freq, step_freq,
        samp_rate, tx_gain,
        receiver_ip="10.153.47.28",  # <-- Replace with receiver's IP
        receiver_port=9999
    )

    while True:
        user_input = input("Press 't' to start transmission or 'q' to quit: ").strip().lower()
        if user_input == 't':
            transmitter.start_transmission()
        elif user_input == 'q':
            print("Exiting.")
            break
        else:
            print("Invalid input. Please press 't' or 'q'.")

