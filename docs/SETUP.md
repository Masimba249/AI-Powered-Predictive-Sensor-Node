# 🛠 Setup Guide

## Hardware Wiring

| STM32 Pin | Peripheral    | Function       |
|-----------|---------------|----------------|
| PB6 (SCL) | MPU-6050, MLX90614, OLED | I2C1 Clock |
| PB7 (SDA) | MPU-6050, MLX90614, OLED | I2C1 Data  |
| PA2 (TX)  | ESP8266 RX    | UART2 TX       |
| PA3 (RX)  | ESP8266 TX    | UART2 RX       |
| PA5       | Green LED     | Status         |
| PC13      | Red LED       | Alert          |
| 3.3V      | All sensors   | Power          |
| GND       | All sensors   | Ground         |

## Firmware Build

1. Install **STM32CubeIDE** v1.14+
2. Clone this repo and open `firmware/` as a CubeMX project
3. Download **TFLite Micro** for ARM Cortex-M and place in `firmware/Middlewares/TFLiteMicro/`
4. Build with: `Release` config (O2 optimisation)
5. Flash via ST-Link: `Run → Debug`

## ML Pipeline

```bash
cd ml_model
pip install -r requirements.txt

# Collect data (run node in DATA_COLLECT mode)
python data_collection/collect_data.py --port /dev/ttyACM0 --label normal --duration 120
python data_collection/collect_data.py --port /dev/ttyACM0 --label bearing_fault --duration 120

# Train + export
python train_model.py

# Evaluate
python evaluate_model.py --model model_int8.tflite
```

Generated `model_data.h` is automatically written to `firmware/`.
Rebuild firmware after model update.

## Dashboard (Node-RED)

```bash
npm install -g node-red
node-red
# Import dashboard/node_red_flow.json via Node-RED UI
```

## MQTT Broker

```bash
sudo apt install mosquitto mosquitto-clients
mosquitto -v
# Subscribe to all topics:
mosquitto_sub -t "predictive-node/#" -v
```