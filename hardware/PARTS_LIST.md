# Parts List - Sign Language Glove MVP

## Core Components

### Microcontroller & Power

| Item                         | Qty | Est. Price | Purpose          | Notes                                 |
| ---------------------------- | --- | ---------- | ---------------- | ------------------------------------- |
| ESP32-S3-DevKitC-1           | 1   | $15-20     | Main processor   | Built-in BLE, USB-C, 240MHz dual-core |
| LiPo Battery 3.7V 2000mAh    | 1   | $10-12     | Power supply     | JST connector, ~4-6 hours runtime     |
| TP4056 Charging Module       | 1   | $3-5       | Battery charging | USB-C or Micro-USB                    |
| LDO Regulator 3.3V (AMS1117) | 1   | $1-2       | Power regulation | 1A output, ESP32 needs stable 3.3V    |
| Slide Switch SPDT            | 1   | $1         | Power on/off     | --                                    |

### Sensors

| Item                              | Qty | Est. Price | Purpose                   | Notes                                              |
| --------------------------------- | --- | ---------- | ------------------------- | -------------------------------------------------- |
| 2.2" Flex Sensor (Spectra Symbol) | 5   | $40-50     | Finger bending            | One per finger (thumb, index, middle, ring, pinky) |
| MPU6050 IMU Module                | 1   | $5-8       | Hand orientation & motion | 6-axis gyro+accel, I2C interface                   |
| Capacitive Touch Sensor (TTP223)  | 5   | $5-8       | Finger contact detection  | Optional but recommended for sign accuracy         |

### Glove & Mounting

| Item                             | Qty     | Est. Price | Purpose             | Notes                            |
| -------------------------------- | ------- | ---------- | ------------------- | -------------------------------- |
| Stretch glove (neoprene/lycra)   | 1       | $5-10      | Base layer          | Snug fit, washable fabric        |
| Velcro strips adhesive           | 1 set   | $3-5       | Sensor mounting     | Removable sensors for washing    |
| Conductive thread/wire 22-26 AWG | 1 spool | $5-8       | Wiring              | Flexible silicone wire preferred |
| Heat shrink tubing kit           | 1       | $5-8       | Wire insulation     | Various sizes                    |
| Small project enclosure          | 1       | $5-8       | Electronics housing | Mounted on back of hand          |

### Connectors & Misc

| Item                               | Qty     | Est. Price | Purpose                    | Notes                  |
| ---------------------------------- | ------- | ---------- | -------------------------- | ---------------------- |
| JST connectors (2-pin, 5-pin sets) | 1 kit   | $8-12      | Modular connections        | For detachable sensors |
| Resistors (10kΩ, 4.7kΩ)            | 10 each | $2-3       | Pull-ups, voltage dividers | --                     |
| Breadboard (optional for testing)  | 1       | $5         | Prototyping                | --                     |
| Jumper wires                       | 1 pack  | $5         | Testing/prototyping        | --                     |

## Total Estimated Cost: **$140-170**

---

## Alternative/Optional Components

### If You Want to Add EMG (Muscle Sensing) - Budget +$50-100

| Item                      | Qty    | Est. Price | Purpose                        |
| ------------------------- | ------ | ---------- | ------------------------------ |
| MyoWare Muscle Sensor     | 2-3    | $40-60     | Detect forearm muscle activity |
| Disposable EMG electrodes | 1 pack | $10-15     | Surface electrodes             |

### Upgrade Options

| Item                                | Est. Price | Benefit                                    |
| ----------------------------------- | ---------- | ------------------------------------------ |
| ESP32-S3 with 8MB PSRAM             | +$5        | More memory for on-device ML               |
| MPU9250 (9-axis) instead of MPU6050 | +$3        | Adds magnetometer for absolute orientation |
| Vibration motor (haptic feedback)   | +$2-3      | User confirmation feedback                 |

---

## Shopping Links (Examples - check availability in your region)

### Amazon/International

- ESP32-S3: Search "ESP32-S3-DevKitC-1" or "ESP32-S3-WROOM"
- Flex Sensors: Search "2.2 inch flex sensor" or "Spectra Symbol flex sensor"
- MPU6050: Search "GY-521 MPU6050"
- TTP223 Touch: Search "TTP223 capacitive touch module"

### AliExpress/Budget Options (slower shipping)

- ESP32-S3: ~$8-12
- Flex sensors: ~$5-7 each (quality may vary)
- MPU6050: ~$2-3
- Complete sensor kits available

### Specialty Electronics (SparkFun, Adafruit, Digi-Key)

- Higher quality but more expensive
- Better documentation and support
- Spectra Symbol flex sensors (gold standard)

---

## What You Might Already Have

- [ ] Breadboard
- [ ] Jumper wires
- [ ] Soldering iron & solder
- [ ] Multimeter
- [ ] USB-C cable
- [ ] Wire stripper/cutter
- [ ] Hot glue gun

---

## Next Steps After Ordering

1. Test each sensor individually with ESP32
2. Verify all connections on breadboard first
3. Calibrate flex sensor ranges
4. Build sensor test firmware
5. Design final glove layout

**Estimated Delivery Time:** 1-2 weeks (depends on supplier)
