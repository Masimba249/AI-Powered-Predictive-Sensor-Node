/**
 * @file  sensor_task.c
 * @brief Reads MPU-6050 (IMU) + MLX90614 (temperature) via I2C.
 *        Accumulates WINDOW_SIZE samples then pushes SensorFrame to queue.
 */

#include "sensor_task.h"
#include "main.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"
#include <string.h>
#include <math.h>

/* ── MPU-6050 register map ─────────────────────────────────────────── */
#define MPU6050_ADDR        0x68 << 1   /* 8-bit HAL address */
#define MPU6050_PWR_MGMT_1  0x6B
#define MPU6050_SMPLRT_DIV  0x19
#define MPU6050_CONFIG      0x1A
#define MPU6050_ACCEL_CFG   0x1C
#define MPU6050_GYRO_CFG    0x1B
#define MPU6050_ACCEL_XOUT  0x3B
#define MPU6050_TEMP_OUT    0x41
#define MPU6050_GYRO_XOUT   0x43

/* Sensitivity scales */
#define ACCEL_SCALE  (9.81f / 16384.0f)   /* ±2g  → m/s² */
#define GYRO_SCALE   (1.0f  / 131.0f)     /* ±250°/s → °/s */

/* ── MLX90614 register map ─────────────────────────────────────────── */
#define MLX90614_ADDR       0x5A << 1
#define MLX90614_TOBJ1      0x07

extern I2C_HandleTypeDef hi2c1;

/* ── Private helpers ───────────────────────────────────────────────── */

static HAL_StatusTypeDef MPU6050_Init(void)
{
    uint8_t data;

    /* Wake up (clear sleep bit) */
    data = 0x00;
    if (HAL_I2C_Mem_Write(&hi2c1, MPU6050_ADDR, MPU6050_PWR_MGMT_1,
                          1, &data, 1, 100) != HAL_OK) return HAL_ERROR;

    /* Sample rate divider → 100 Hz: 1kHz / (1+9) = 100Hz */
    data = 0x09;
    HAL_I2C_Mem_Write(&hi2c1, MPU6050_ADDR, MPU6050_SMPLRT_DIV, 1, &data, 1, 100);

    /* DLPF bandwidth 44 Hz */
    data = 0x03;
    HAL_I2C_Mem_Write(&hi2c1, MPU6050_ADDR, MPU6050_CONFIG, 1, &data, 1, 100);

    /* Accel ±2g */
    data = 0x00;
    HAL_I2C_Mem_Write(&hi2c1, MPU6050_ADDR, MPU6050_ACCEL_CFG, 1, &data, 1, 100);

    /* Gyro ±250°/s */
    data = 0x00;
    HAL_I2C_Mem_Write(&hi2c1, MPU6050_ADDR, MPU6050_GYRO_CFG, 1, &data, 1, 100);

    return HAL_OK;
}

static HAL_StatusTypeDef MPU6050_ReadSample(SensorSample_t *s)
{
    uint8_t buf[14];

    /* Burst read: ACCEL(6) + TEMP(2) + GYRO(6) */
    if (HAL_I2C_Mem_Read(&hi2c1, MPU6050_ADDR, MPU6050_ACCEL_XOUT,
                         1, buf, 14, 50) != HAL_OK) {
        return HAL_ERROR;
    }

    int16_t raw_ax = (int16_t)((buf[0]  << 8) | buf[1]);
    int16_t raw_ay = (int16_t)((buf[2]  << 8) | buf[3]);
    int16_t raw_az = (int16_t)((buf[4]  << 8) | buf[5]);
    int16_t raw_gx = (int16_t)((buf[8]  << 8) | buf[9]);
    int16_t raw_gy = (int16_t)((buf[10] << 8) | buf[11]);
    int16_t raw_gz = (int16_t)((buf[12] << 8) | buf[13]);

    s->accel_x = raw_ax * ACCEL_SCALE;
    s->accel_y = raw_ay * ACCEL_SCALE;
    s->accel_z = raw_az * ACCEL_SCALE;
    s->gyro_x  = raw_gx * GYRO_SCALE;
    s->gyro_y  = raw_gy * GYRO_SCALE;
    s->gyro_z  = raw_gz * GYRO_SCALE;

    s->timestamp_ms = xTaskGetTickCount();
    return HAL_OK;
}

static float MLX90614_ReadTemp(void)
{
    uint8_t buf[3];
    float temp_k;

    if (HAL_I2C_Mem_Read(&hi2c1, MLX90614_ADDR, MLX90614_TOBJ1,
                         1, buf, 3, 50) != HAL_OK) {
        return -273.15f; /* error sentinel */
    }

    /* 16-bit PEC format: [LSB, MSB, PEC] */
    uint16_t raw = ((uint16_t)buf[1] << 8) | buf[0];
    temp_k = raw * 0.02f;           /* 0.02 K per LSB */
    return temp_k - 273.15f;        /* Convert to °C  */
}

/* ── Sensor Task ───────────────────────────────────────────────────── */
void vSensorTask(void *pvParameters)
{
    (void)pvParameters;

    SensorFrame_t  frame;
    static uint32_t window_id = 0;

    /* Init sensors with I2C mutex */
    xSemaphoreTake(xI2CMutex, portMAX_DELAY);
    MPU6050_Init();
    xSemaphoreGive(xI2CMutex);

    /* Wait 200ms for sensors to settle */
    vTaskDelay(pdMS_TO_TICKS(200));

    memset(&frame, 0, sizeof(SensorFrame_t));
    uint8_t idx = 0;

    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xPeriod = pdMS_TO_TICKS(10); /* 100 Hz */

    for (;;) {
        vTaskDelayUntil(&xLastWakeTime, xPeriod);

        xSemaphoreTake(xI2CMutex, portMAX_DELAY);
        HAL_StatusTypeDef imu_ok  = MPU6050_ReadSample(&frame.samples[idx]);
        float temp = MLX90614_ReadTemp();
        xSemaphoreGive(xI2CMutex);

        if (imu_ok == HAL_OK) {
            frame.samples[idx].temperature = temp;
            idx++;
        }

        if (idx >= WINDOW_SIZE) {
            frame.sample_count = WINDOW_SIZE;
            frame.window_id    = window_id++;
            idx = 0;

            /* Non-blocking send — drop frame if queue full */
            xQueueSend(xSensorDataQueue, &frame, 0);
        }
    }
}