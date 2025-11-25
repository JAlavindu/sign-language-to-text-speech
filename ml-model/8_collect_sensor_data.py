"""
Sensor Data Collection Script
Connects to ESP32 Glove via BLE and records sensor data for training.
"""

import asyncio
import struct
import time
import os
import csv
from bleak import BleakClient, BleakScanner

# Configuration
DEVICE_NAME = "ASL_Glove_001"
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
CHARACTERISTIC_UUID_TX = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
DATA_DIR = os.path.join("ml-model", "datasets", "sensor_data", "raw")

# Global state
is_recording = False
current_label = ""
recorded_data = []
packet_count = 0

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_packet(data):
    """
    Parses the binary packet from ESP32
    Struct format: I (timestamp 4) + 5H (flex 10) + 3f (accel 12) + 3f (gyro 12) + 5B (touch 5) = 43 bytes
    Note: Arduino struct padding might affect size. Using standard packing.
    """
    try:
        # Unpack: I=uint32, H=uint16, f=float, B=uint8
        # Total size = 4 + 10 + 12 + 12 + 5 = 43 bytes
        # Note: Check your ESP32 struct packing! This assumes packed or matching alignment.
        # If ESP32 aligns to 4 bytes, size might be different.
        
        # Let's assume standard packing for now. 
        # If data length doesn't match, we might need to adjust format string.
        if len(data) != 43:
            # print(f"Warning: Unexpected packet size {len(data)}")
            return None

        unpacked = struct.unpack('<I5H3f3f5B', data)
        
        return {
            'timestamp': unpacked[0],
            'flex': unpacked[1:6],
            'accel': unpacked[6:9],
            'gyro': unpacked[9:12],
            'touch': unpacked[12:17]
        }
    except Exception as e:
        print(f"Parse error: {e}")
        return None

def notification_handler(sender, data):
    global is_recording, recorded_data, packet_count
    
    packet = parse_packet(data)
    if packet:
        packet_count += 1
        
        # Print status every 10 packets
        if packet_count % 10 == 0:
            print(f"\rFlex: {packet['flex']} | Accel: {packet['accel'][0]:.2f}", end="")

        if is_recording:
            row = [packet['timestamp']] + \
                  list(packet['flex']) + \
                  list(packet['accel']) + \
                  list(packet['gyro']) + \
                  list(packet['touch']) + \
                  [current_label]
            recorded_data.append(row)

async def main():
    global is_recording, current_label, recorded_data
    
    ensure_dir(DATA_DIR)
    
    print(f"Scanning for {DEVICE_NAME}...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME)
    
    if not device:
        print(f"Device '{DEVICE_NAME}' not found. Make sure ESP32 is on and advertising.")
        return

    print(f"Connecting to {device.address}...")
    async with BleakClient(device) as client:
        print("Connected!")
        
        await client.start_notify(CHARACTERISTIC_UUID_TX, notification_handler)
        
        print("\nControls:")
        print("  [R] - Start/Stop Recording")
        print("  [Q] - Quit")
        
        while True:
            cmd = await asyncio.to_thread(input, "\nEnter command (r/q): ")
            
            if cmd.lower() == 'q':
                break
            
            elif cmd.lower() == 'r':
                if not is_recording:
                    current_label = input("Enter label for this sign (e.g., A): ").upper()
                    if not current_label:
                        print("Label required!")
                        continue
                        
                    print(f"Recording '{current_label}'... (Press Enter to stop)")
                    recorded_data = []
                    is_recording = True
                    
                    # Wait for user to press enter to stop
                    await asyncio.to_thread(input, "")
                    
                    is_recording = False
                    print(f"Stopped. Captured {len(recorded_data)} samples.")
                    
                    # Save to CSV
                    filename = os.path.join(DATA_DIR, f"{current_label}_{int(time.time())}.csv")
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['timestamp', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5', 
                                       'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z',
                                       'touch1', 'touch2', 'touch3', 'touch4', 'touch5', 'label'])
                        writer.writerows(recorded_data)
                    print(f"Saved to {filename}")
                
        await client.stop_notify(CHARACTERISTIC_UUID_TX)

if __name__ == "__main__":
    asyncio.run(main())
