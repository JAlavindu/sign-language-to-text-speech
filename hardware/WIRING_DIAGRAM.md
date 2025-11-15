# Wiring Diagram & Pin Assignments

## ESP32-S3 Pin Configuration

### Power Pins

- **3.3V** → All sensor VCC
- **GND** → All sensor GND (common ground)
- **5V** → Battery input (through LDO regulator to 3.3V)

### Flex Sensors (Analog Inputs)

Each flex sensor is wired as a voltage divider with a 10kΩ resistor:

```
3.3V ──[Flex Sensor]──┬──[10kΩ]── GND
                      │
                      └─→ ADC Pin
```

| Finger | Flex Sensor | ESP32-S3 Pin | ADC Channel |
| ------ | ----------- | ------------ | ----------- |
| Thumb  | Flex 1      | GPIO 1       | ADC1_CH0    |
| Index  | Flex 2      | GPIO 2       | ADC1_CH1    |
| Middle | Flex 3      | GPIO 3       | ADC1_CH2    |
| Ring   | Flex 4      | GPIO 4       | ADC1_CH3    |
| Pinky  | Flex 5      | GPIO 5       | ADC1_CH4    |

**Note:** ESP32-S3 has 12-bit ADC (0-4095 values)

### MPU6050 IMU (I2C Interface)

| MPU6050 Pin | ESP32-S3 Pin | Description          |
| ----------- | ------------ | -------------------- |
| VCC         | 3.3V         | Power                |
| GND         | GND          | Ground               |
| SCL         | GPIO 9       | I2C Clock            |
| SDA         | GPIO 8       | I2C Data             |
| INT         | GPIO 10      | Interrupt (optional) |

**I2C Address:** 0x68 (default) or 0x69 (if AD0 pulled high)

### Capacitive Touch Sensors (TTP223)

Digital pins configured as inputs:

| Finger | Touch Sensor | ESP32-S3 Pin | Description |
| ------ | ------------ | ------------ | ----------- |
| Thumb  | Touch 1      | GPIO 11      | Thumb tip   |
| Index  | Touch 2      | GPIO 12      | Index tip   |
| Middle | Touch 3      | GPIO 13      | Middle tip  |
| Ring   | Touch 4      | GPIO 14      | Ring tip    |
| Pinky  | Touch 5      | GPIO 15      | Pinky tip   |

**Touch Logic:** HIGH (3.3V) when touched, LOW (0V) when not touched

### Additional Peripherals

| Component               | ESP32-S3 Pin | Description                |
| ----------------------- | ------------ | -------------------------- |
| Power Switch            | EN pin       | Pull LOW to turn off       |
| Status LED              | GPIO 38      | Onboard LED for status     |
| Haptic Motor (optional) | GPIO 16      | PWM for vibration feedback |

---

## Circuit Diagram (Text-based)

```
                    ┌─────────────────────┐
                    │    ESP32-S3-S3      │
                    │                     │
    Battery ────────┤ 5V            3.3V  ├──┬─── All Sensor VCC
    (3.7V LiPo)     │                     │  │
                    │ GND            GND  ├──┴─── Common Ground
                    │                     │
    Flex Sensors:   │                     │
    ────────────────┤ GPIO1  (ADC1_0)     │ Thumb
    ────────────────┤ GPIO2  (ADC1_1)     │ Index
    ────────────────┤ GPIO3  (ADC1_2)     │ Middle
    ────────────────┤ GPIO4  (ADC1_3)     │ Ring
    ────────────────┤ GPIO5  (ADC1_4)     │ Pinky
                    │                     │
    MPU6050 IMU:    │                     │
    ────────────────┤ GPIO8  (SDA)        │ I2C Data
    ────────────────┤ GPIO9  (SCL)        │ I2C Clock
    ────────────────┤ GPIO10 (INT)        │ Interrupt
                    │                     │
    Touch Sensors:  │                     │
    ────────────────┤ GPIO11              │ Thumb touch
    ────────────────┤ GPIO12              │ Index touch
    ────────────────┤ GPIO13              │ Middle touch
    ────────────────┤ GPIO14              │ Ring touch
    ────────────────┤ GPIO15              │ Pinky touch
                    │                     │
    Optional:       │                     │
    ────────────────┤ GPIO16 (PWM)        │ Haptic motor
    ────────────────┤ GPIO38              │ Status LED
                    │                     │
                    │ USB-C               │ Programming & Debug
                    └─────────────────────┘
```

---

## Flex Sensor Voltage Divider Details

Each flex sensor needs a pull-down resistor to create a voltage divider:

```
        3.3V
         │
         ├─── Flex Sensor (Resistance changes: 10kΩ flat → 30-40kΩ bent)
         │
         ├───┬─── To ESP32 ADC pin
         │   │
        10kΩ │ (Pull-down resistor)
         │   │
        GND ─┘
```

**Voltage Reading:**

- **Flat (10kΩ):** V = 3.3V × 10kΩ/(10kΩ+10kΩ) = 1.65V → ADC ~2048
- **Bent (40kΩ):** V = 3.3V × 10kΩ/(40kΩ+10kΩ) = 0.66V → ADC ~820

**Calibration Range:**

- Straight finger: ~1800-2200 (will vary per sensor)
- Fully bent: ~600-1000

---

## Physical Sensor Placement on Glove

### Top View of Right Hand Glove:

```
        [Pinky]   [Ring]   [Middle]  [Index]   [Thumb]
          │         │         │         │         │
          ●─────────●─────────●─────────●─────────●  ← Touch sensors (fingertips)
          │         │         │         │         │
          ║         ║         ║         ║         ║  ← Flex sensors (along fingers)
          ║         ║         ║         ║         ║
          ╚═════════╩═════════╩═════════╩═════════╝
                            │
                         [■ MPU6050]  ← IMU on back of hand
                            │
                         [▣ ESP32]    ← Electronics enclosure
                            │
                         [🔋 Battery]  ← LiPo on wrist strap
```

### Detailed Placement:

1. **Flex Sensors:**

   - Run along the **back** of each finger (top side, not palm)
   - Start at knuckle, end before fingertip
   - Secure with fabric glue or sewn pockets
   - Keep flat against finger for accurate readings

2. **Touch Sensors (TTP223 modules):**

   - Mount on **fingertips** (pad side)
   - Small conductive pads on thumb, index, middle (primary contact points)
   - Can skip ring/pinky for MVP if space-limited

3. **MPU6050 IMU:**

   - Mount on **back of hand** (between wrist and knuckles)
   - Orient with axes aligned to hand coordinate system
   - Secure firmly (any movement causes noise)

4. **ESP32-S3:**

   - Small enclosure on **back of hand** or **wrist strap**
   - Keep accessible for USB programming
   - Ensure antenna (BLE) is not blocked

5. **Battery:**
   - Wrist strap or forearm band
   - Connect via JST connector for easy removal
   - Weight distribution: keep battery separate from glove

---

## Assembly Steps

### Step 1: Breadboard Prototype (Test First!)

Before mounting on glove, test everything on breadboard:

1. Connect one flex sensor → verify readings
2. Connect MPU6050 → verify I2C communication
3. Connect one touch sensor → verify digital input
4. Test all sensors together → check for conflicts

### Step 2: Create Wiring Harness

1. Cut wires to appropriate lengths (measure on your hand)
2. Solder connections to sensors
3. Use heat shrink tubing for insulation
4. Bundle wires together with cable sleeves
5. Add JST connectors for detachable sensors

### Step 3: Mount on Glove

1. Mark sensor positions on glove
2. Sew or glue sensor pockets/strips
3. Route wires along fingers → back of hand
4. Secure wires with fabric tape or stitching
5. Mount electronics enclosure with Velcro

### Step 4: Power System

1. Solder battery to TP4056 charging module
2. Connect LDO regulator output to ESP32 VIN/3.3V
3. Add power switch in series with battery
4. Test voltage levels with multimeter

---

## Testing Checklist

Before final assembly:

- [ ] All flex sensors read values (1000-2500 range)
- [ ] MPU6050 detected on I2C (address 0x68)
- [ ] IMU returns accelerometer/gyro values
- [ ] Touch sensors trigger HIGH when pressed
- [ ] No short circuits (check with multimeter)
- [ ] Battery charges properly via USB
- [ ] ESP32 boots and runs test code
- [ ] BLE advertising works
- [ ] All wires secured and insulated

---

## Common Issues & Troubleshooting

| Problem                  | Possible Cause            | Solution                                   |
| ------------------------ | ------------------------- | ------------------------------------------ |
| No flex sensor reading   | No pull-down resistor     | Add 10kΩ to GND                            |
| Noisy flex values        | Poor connection           | Re-solder, check continuity                |
| MPU6050 not detected     | Wrong I2C pins or address | Check GPIO8/9, try address 0x69            |
| Touch sensor always HIGH | Wiring issue              | Check VCC/GND, sensor orientation          |
| ESP32 won't boot         | Power brownout            | Check battery voltage, add capacitor       |
| BLE won't connect        | Antenna blocked           | Reposition ESP32, check metal interference |

---

## Safety Notes

⚠️ **Important:**

- Never connect battery backwards (check polarity!)
- Use proper gauge wire for current capacity
- Insulate all solder joints with heat shrink
- Test with multimeter before powering on
- Keep battery away from sharp objects
- Don't charge unattended

---

## Next: Test Each Sensor Individually

After wiring, we'll write simple test code for:

1. Reading one flex sensor
2. Reading MPU6050 accelerometer/gyro
3. Reading touch sensors
4. Combining all sensor data

Ready to move to Step 4 (Development Environment Setup)?
