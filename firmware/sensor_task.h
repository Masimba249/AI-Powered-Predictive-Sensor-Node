#ifndef SENSOR_TASK_H
#define SENSOR_TASK_H

#include <stdint.h>
#include "FreeRTOS.h"
#include "queue.h"

/* Number of samples per inference window (1 second @ 100 Hz) */
#define WINDOW_SIZE         100
#define SENSOR_AXES         3    /* X, Y, Z acceleration */

/**
 * @brief One sample from both sensors.
 */
typedef struct {
    float accel_x;   /* m/s²  */
    float accel_y;
    float accel_z;
    float gyro_x;    /* °/s   */
    float gyro_y;
    float gyro_z;
    float temperature; /* °C  */
    uint32_t timestamp_ms;
} SensorSample_t;

/**
 * @brief One full inference window (100 samples).
 */
typedef struct {
    SensorSample_t samples[WINDOW_SIZE];
    uint8_t        sample_count;
    uint32_t       window_id;
} SensorFrame_t;

/**
 * @brief Alert message sent from InferenceTask → CommsTask.
 */
typedef struct {
    uint8_t  anomaly_detected;   /* 1 = fault, 0 = normal        */
    float    confidence;         /* Model output probability      */
    uint8_t  fault_class;        /* 0=normal,1=bearing,2=imbalance*/
    float    temperature;        /* Latest temperature reading    */
    uint32_t window_id;
} AlertMessage_t;

/* Label strings for fault classes */
static const char* const FAULT_LABELS[] = {
    "NORMAL",
    "BEARING_FAULT",
    "IMBALANCE",
    "LOOSENESS"
};
#define NUM_FAULT_CLASSES 4

/* External RTOS handles (defined in main.c) */
extern QueueHandle_t   xSensorDataQueue;
extern SemaphoreHandle_t xI2CMutex;

/**
 * @brief FreeRTOS sensor acquisition task.
 *        Collects WINDOW_SIZE samples and posts a SensorFrame to the queue.
 */
void vSensorTask(void *pvParameters);

#endif /* SENSOR_TASK_H */