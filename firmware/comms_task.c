/**
 * @file  comms_task.c
 * @brief Sends alert/telemetry data over UART to ESP8266 WiFi module.
 *        ESP8266 bridges to MQTT broker (e.g., Mosquitto / AWS IoT).
 *
 * AT command protocol:
 *   STM32 sends JSON payload over UART2 → ESP8266 publishes to MQTT topic.
 *
 * MQTT Topics:
 *   predictive-node/telemetry  — every window (JSON)
 *   predictive-node/alert      — anomaly only (JSON)
 */

#include "comms_task.h"
#include "sensor_task.h"
#include "main.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include <stdio.h>
#include <string.h>

extern UART_HandleTypeDef huart2;
extern QueueHandle_t      xAlertQueue;

#define UART_TX_TIMEOUT_MS 1000
#define TX_BUF_SIZE        256

/* Heartbeat every N windows even without alerts */
#define HEARTBEAT_EVERY_N_WINDOWS 10

static void uart_send(const char *str)
{
    HAL_UART_Transmit(&huart2, (uint8_t *)str, strlen(str),
                      UART_TX_TIMEOUT_MS);
}

/**
 * @brief Build and send a JSON telemetry message via UART.
 *        Format: {"wid":<id>,"cls":<class>,"lbl":"<label>","conf":<0.xx>,"temp":<xx.x>,"alert":<0|1>}
 */
static void send_mqtt_payload(const AlertMessage_t *alert)
{
    char buf[TX_BUF_SIZE];
    const char *label = (alert->fault_class < NUM_FAULT_CLASSES)
                        ? FAULT_LABELS[alert->fault_class]
                        : "UNKNOWN";

    int len = snprintf(buf, TX_BUF_SIZE,
        "MQTT:{\"wid\":%lu,\"cls\":%u,\"lbl\":\"%s\","
        "\"conf\":%.3f,\"temp\":%.1f,\"alert\":%u}\r\n",
        (unsigned long)alert->window_id,
        alert->fault_class,
        label,
        alert->confidence,
        alert->temperature,
        alert->anomaly_detected);

    if (len > 0 && len < TX_BUF_SIZE) {
        uart_send(buf);
    }

    /* Toggle alert LED */
    if (alert->anomaly_detected) {
        HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_RESET); /* Active LOW */
    } else {
        HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_SET);
    }
}

void vCommsTask(void *pvParameters)
{
    (void)pvParameters;

    AlertMessage_t alert;
    static uint32_t last_window_sent = 0;

    /* Wait for ESP8266 to boot */
    vTaskDelay(pdMS_TO_TICKS(3000));

    /* Init AT command handshake */
    uart_send("AT+RST\r\n");
    vTaskDelay(pdMS_TO_TICKS(2000));
    uart_send("AT+CWMODE=1\r\n");
    vTaskDelay(pdMS_TO_TICKS(500));
    uart_send("AT+CWJAP=\"YOUR_SSID\",\"YOUR_PASS\"\r\n");
    vTaskDelay(pdMS_TO_TICKS(5000));

    for (;;) {
        if (xQueueReceive(xAlertQueue, &alert, pdMS_TO_TICKS(1000)) == pdTRUE) {
            send_mqtt_payload(&alert);
            last_window_sent = alert.window_id;
        } else {
            /* Heartbeat: send a keep-alive every ~10s if no data */
            AlertMessage_t hb = {0};
            hb.fault_class = 0;
            hb.confidence  = 1.0f;
            hb.anomaly_detected = 0;
            hb.window_id = last_window_sent;
            send_mqtt_payload(&hb);
        }
    }
}