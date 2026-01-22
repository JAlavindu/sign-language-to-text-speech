# 🧤 Smart Glove Assembly Guide

This guide walks you through the physical construction and wiring of the Sign Language Glove. It covers how to assemble the components listed in the `hardware/PARTS_LIST.md` and connect them according to the `hardware/WIRING_DIAGRAM.md`.

## 📋 Prerequisites

### Tools Needed
*   Soldering Iron & Solder
*   Wire Strippers/Cutters
*   Hot Glue Gun or Fabric Glue
*   Needle & Thread (if sewing sensors)
*   Multimeter (for troubleshooting)

### Core Components
*   **ESP32-S3 Board** (The brain)
*   **5x Flex Sensors** (The fingers)
*   **MPU6050 IMU** (The orientation)
*   **Resistors (10kΩ)** (For flex sensors)
*   **Glove** (Base material)
*   **Wires & Cables**

---

## ⚡ Step 1: The Circuit Concepts

Before gluing anything to the glove, it is highly recommended to build a prototype on a breadboard or solder the sub-assemblies.

### 1. Flex Sensors (Voltage Finders)
Flex sensors change resistance when bent. The ESP32 cannot read resistance directly, only voltage. We use a **Voltage Divider** circuit to convert this resistance change into a voltage change (0V - 3.3V).

**The Circuit per Finger:**
```
[3.3V] ──── [Flex Sensor] ────┬──── [10kΩ Resistor] ──── [GND]
                              │
                              └──── [To ESP32 GPIO Pin]
```
*   **Why 10kΩ?** It balances the typical resistance range of the flex sensor (usually 10k-30kΩ) to give a good voltage swing.

### 2. IMU (Motion Sensor)
The MPU6050 communicates via I2C. It needs just 4 wires:
*   **VCC** -> 3.3V
*   **GND** -> GND
*   **SDA** -> Data Line
*   **SCL** -> Clock Line

---

## 🛠️ Step 2: Physical Assembly

### 1. Preparing the Glove
1.  Put on the glove and mark the positions of your knuckles and finger joints with a marker.
2.  The flex sensors should sit **on top** of the fingers, bridging across the knuckles.

### 2. Mounting Flex Sensors
*   **Placement:** Align each sensor so the bendable area covers the main knuckle.
*   **Securing:**
    *   *Option A (Sewing):* Sew small loops of thread over the non-active parts of the sensor (the ends) to hold it in place. **Do not puncture the sensor!**
    *   *Option B (Sleeves):* Sew small fabric pockets/sleeves onto the glove fingers and slide the sensors in. This allows for easy replacement.
*   **Routing:** Run the wires from the base of each finger towards the back of the wrist.

### 3. Mounting the Brain (ESP32 + Battery)
1.  Attach the **ESP32** and **MPU6050** to the back of the hand (dorsum).
    *   You can use a small plastic enclosure or a 3D printed mount.
    *   Alternatively, sew a velcro patch to the back of the glove and attach the electronics with velcro.
2.  **Orientation matters!** Mount the MPU6050 flat on the back of the hand. Note which direction is "Forward".

### 4. Wiring It Up
Use flexible silicone wire (26-28 AWG) if possible, as it moves better with your hand.

#### Connecting Flex Sensors to ESP32
Refer to `hardware/WIRING_DIAGRAM.md` for exact pin numbers, but generally:
1.  Connect one leg of **ALL** 5 flex sensors together. Connect this to **3.3V**.
2.  Connect the other leg of each sensor to its own **10kΩ resistor**.
3.  Connect the other side of **ALL** 5 resistors together. Connect this to **GND**.
4.  Run a wire from the point *between* the sensor and resistor (the "Tap" point) to the ESP32 GPIO pin.

| Finger | ESP32-S3 GPIO Suggestion |
| :--- | :--- |
| Thumb | GPIO 1 |
| Index | GPIO 2 |
| Middle | GPIO 3 |
| Ring | GPIO 4 |
| Pinky | GPIO 5 |

#### Connecting the IMU
1.  **VCC** → ESP32 3.3V
2.  **GND** → ESP32 GND
3.  **SDA** → GPIO 8 (or your board's SDA pin)
4.  **SCL** → GPIO 9 (or your board's SCL pin)

---

## 🔍 Step 3: Testing Before Use

1.  **Continuity Check:** Use a multimeter to ensure no shorts between 3.3V and GND.
2.  **Power Up:** Plug in the ESP32 via USB.
3.  **Upload Firmware:** Open `firmware/sensor_streamer/sensor_streamer.ino` and upload it.
4.  **Verify Data:** Open the Serial Monitor (115200 baud).
    *   You should see a stream of JSON data: `{"flex": [...], "imu": [...]}`.
    *   Bend a finger; the corresponding value in the `flex` array should change.
    *   Rotate your hand; the `imu` values should change.

## ⚠️ Important Notes

*   **Strain Relief:** Hot glue the wire connections at the sensor terminals. These are fragile and break easily with movement.
*   **Cable Slack:** Leave enough wire slack around the wrist so you can flex your hand fully without pulling cables tight.
*   **Isolation:** Ensure the conductive parts of the sensors (pins) don't touch your skin (sweat is conductive!). Wrap connections in heat shrink.
