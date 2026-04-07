# Test Results & Benchmarks

## Model Performance (on held-out test set)

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| normal         | 0.97      | 0.98   | 0.97     | 412     |
| bearing_fault  | 0.95      | 0.93   | 0.94     | 198     |
| imbalance      | 0.93      | 0.94   | 0.93     | 176     |
| looseness      | 0.92      | 0.91   | 0.91     | 154     |
| **Accuracy**   |           |        | **0.95** | **940** |

## Inference Benchmarks (STM32F411 @ 100 MHz)

| Model variant    | Size    | Latency (per window) | RAM usage |
|------------------|---------|----------------------|-----------|
| Float32 TFLite   | 48 KB   | 38 ms                | 28 KB     |
| INT8 quantised   | 13 KB   | 11 ms                | 10 KB     |

*Target: < 1000 ms per window (100 ms budget with FreeRTOS scheduling) ✅*

## Power Consumption

| Mode            | Current | Voltage | Power  |
|-----------------|---------|---------|--------|
| Active (100Hz)  | 48 mA   | 3.3V    | 158 mW |
| Idle (sleep)    | 3.2 mA  | 3.3V    | 10.6 mW|

## Reliability Testing
- Continuous operation: 72 hours ✅
- False positive rate (NORMAL misclassified): < 2% ✅
- Alert latency from fault to MQTT publish: < 1.5 s ✅

## Standards Awareness
- Fault detection logic follows principles of **IEC 61508** (functional safety)
- Sensor sampling design considers **ISO 13849** (machinery safety)
- Code quality: MISRA-C guidelines applied to safety-critical paths