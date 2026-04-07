# 🤖 AI-Powered Predictive Maintenance Sensor Node
### TinyML + STM32 + Industry 4.0 | Portfolio Project

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Platform: STM32](https://img.shields.io/badge/platform-STM32F4-blue)]()
[![ML: TFLite Micro](https://img.shields.io/badge/ML-TFLite%20Micro-orange)]()

A low-power embedded sensor node that uses a trained TinyML model running **on-device** (STM32 ARM Cortex-M4) to detect early signs of motor/bearing failure by analyzing vibration and temperature data — **no cloud required**.

---

## 🧱 Architecture

```
[MPU-6050 IMU] ──┐
                 ├──► [STM32F4 + FreeRTOS] ──► [TFLite Micro Inference] ──► [BLE/MQTT Alert]
[MLX90614 Temp]──┘         │
                           └──► [OLED Display] + [Local Logging]
```

## 🔧 Hardware
- STM32F411 Nucleo (ARM Cortex-M4 @ 100MHz)
- MPU-6050 (6-axis IMU via I2C)
- MLX90614 (IR Temperature Sensor via I2C)
- ESP8266 / nRF52 (BLE or WiFi reporting via UART)
- 1.3" SSD1306 OLED Display
- LiPo Battery + TP4056 Charger

## 💻 Software Stack
- STM32CubeIDE + HAL Drivers (C/C++)
- FreeRTOS v10
- TensorFlow Lite for Microcontrollers
- MQTT over ESP8266 (WiFi bridge)
- Python 3.9 (model training pipeline)
- Edge Impulse compatible dataset format

## 📁 Project Structure
```
predictive-maintenance-node/
├── firmware/           # STM32 HAL + FreeRTOS C firmware
├── ml_model/           # Python training pipeline + TFLite model
├── dashboard/          # Node-RED flow for visualization
├── docs/               # Schematics, architecture, test results
└── README.md
```

## 🚀 Getting Started
See [docs/SETUP.md](docs/SETUP.md) for full setup instructions.

---

*Dieses Projekt demonstriert TinyML auf eingebetteten ARM-Systemen für industrielle Zustandsüberwachung.*
*(This project demonstrates TinyML on embedded ARM systems for industrial condition monitoring.)*