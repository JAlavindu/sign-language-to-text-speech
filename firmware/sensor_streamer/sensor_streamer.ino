/*
  ASL Glove Sensor Streamer
  Streams Flex, IMU, and Touch data via BLE
*/

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// BLE UUIDs
#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E" // UART Service
#define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

// Pins
const int FLEX_PINS[] = {36, 39, 34, 35, 32}; // VP, VN, 34, 35, 32 (ESP32 ADC pins)
const int TOUCH_PINS[] = {T0, T3, T4, T5, T6}; // Touch pins (GPIO 4, 15, 13, 12, 14)

// Globals
BLEServer* pServer = NULL;
BLECharacteristic* pTxCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;
Adafruit_MPU6050 mpu;

// Data Packet Structure
struct SensorPacket {
  uint32_t timestamp;
  uint16_t flex[5];
  float accel[3];
  float gyro[3];
  uint8_t touch[5];
};

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
    };

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
    }
};

void setup() {
  Serial.begin(115200);

  // Init IMU
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) { delay(10); }
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // Init BLE
  BLEDevice::init("ASL_Glove_001");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());
  BLEService *pService = pServer->createService(SERVICE_UUID);
  pTxCharacteristic = pService->createCharacteristic(
                    CHARACTERISTIC_UUID_TX,
                    BLECharacteristic::PROPERTY_NOTIFY
                  );
  pTxCharacteristic->addDescriptor(new BLE2902());
  pService->start();
  pServer->getAdvertising()->start();
  Serial.println("Waiting for client connection...");
}

void loop() {
  if (deviceConnected) {
    SensorPacket packet;
    packet.timestamp = millis();

    // Read Flex Sensors
    for (int i = 0; i < 5; i++) {
      packet.flex[i] = analogRead(FLEX_PINS[i]);
    }

    // Read IMU
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    packet.accel[0] = a.acceleration.x;
    packet.accel[1] = a.acceleration.y;
    packet.accel[2] = a.acceleration.z;
    packet.gyro[0] = g.gyro.x;
    packet.gyro[1] = g.gyro.y;
    packet.gyro[2] = g.gyro.z;

    // Read Touch
    for (int i = 0; i < 5; i++) {
      packet.touch[i] = (touchRead(TOUCH_PINS[i]) < 40) ? 1 : 0; // Threshold check
    }

    // Send Data
    pTxCharacteristic->setValue((uint8_t*)&packet, sizeof(SensorPacket));
    pTxCharacteristic->notify();
    
    delay(10); // ~100Hz
  }

  // Disconnect handling
  if (!deviceConnected && oldDeviceConnected) {
      delay(500); 
      pServer->startAdvertising(); 
      Serial.println("Start advertising");
      oldDeviceConnected = deviceConnected;
  }
  if (deviceConnected && !oldDeviceConnected) {
      oldDeviceConnected = deviceConnected;
  }
}
