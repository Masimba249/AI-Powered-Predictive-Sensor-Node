/**
 * @file  display_task.c
 * @brief Updates a 128×64 SSD1306 OLED display via I2C.
 *        Shows: status, last fault class, confidence, temperature, uptime.
 */

#include "display_task.h"
#include "sensor_task.h"
#include "main.h"
#include "ssd1306.h"       /* Lightweight SSD1306 HAL driver */
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include <stdio.h>
#include <string.h>

extern QueueHandle_t     xAlertQueue;
extern SemaphoreHandle_t xI2CMutex;

void vDisplayTask(void *pvParameters)
{
    (void)pvParameters;

    xSemaphoreTake(xI2CMutex, portMAX_DELAY);
    SSD1306_Init();
    xSemaphoreGive(xI2CMutex);

    AlertMessage_t latest = {0};
    latest.fault_class = 0;
    latest.confidence  = 0.0f;

    char line[32];
    uint32_t uptime_s = 0;

    for (;;) {
        /* Peek at latest alert (non-blocking) */
        xQueuePeek(xAlertQueue, &latest, 0);

        xSemaphoreTake(xI2CMutex, portMAX_DELAY);

        SSD1306_Clear();

        /* Line 0: Title */
        SSD1306_GotoXY(0, 0);
        SSD1306_Puts("PredMaint Node", &Font_7x10, 1);

        /* Line 1: Status */
        SSD1306_GotoXY(0, 14);
        if (latest.anomaly_detected) {
            snprintf(line, sizeof(line), "! %s",
                     FAULT_LABELS[latest.fault_class]);
        } else {
            snprintf(line, sizeof(line), "Status: NORMAL");
        }
        SSD1306_Puts(line, &Font_7x10, 1);

        /* Line 2: Confidence */
        SSD1306_GotoXY(0, 28);
        snprintf(line, sizeof(line), "Conf: %.1f%%", latest.confidence * 100.0f);
        SSD1306_Puts(line, &Font_7x10, 1);

        /* Line 3: Temperature */
        SSD1306_GotoXY(0, 42);
        snprintf(line, sizeof(line), "Temp: %.1f C", latest.temperature);
        SSD1306_Puts(line, &Font_7x10, 1);

        /* Line 4: Uptime */
        SSD1306_GotoXY(0, 56);
        snprintf(line, sizeof(line), "Up: %lus  W:%lu",
                 (unsigned long)uptime_s,
                 (unsigned long)latest.window_id);
        SSD1306_Puts(line, &Font_7x10, 1);

        SSD1306_UpdateScreen();
        xSemaphoreGive(xI2CMutex);

        uptime_s++;
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}