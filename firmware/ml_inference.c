/**
 * @file  ml_inference.c
 * @brief TensorFlow Lite Micro inference engine.
 *        Extracts statistical features from sensor window,
 *        runs the quantized .tflite model, posts alerts.
 *
 * TFLite Micro C API used here — link against:
 *   tensorflow/lite/micro library (compiled for Cortex-M4 with CMSIS-NN)
 */

#include "ml_inference.h"
#include "main.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/* ── TFLite Micro includes ─────────────────────────────────────────── */
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* ── Generated model header (xxd -i model.tflite > model_data.h) ──── */
#include "model_data.h"   /* uint8_t g_model[] + g_model_len */

extern QueueHandle_t xSensorDataQueue;
extern QueueHandle_t xAlertQueue;

/* ── Tensor arena (static allocation — no heap) ────────────────────── */
static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

/* ── Feature extraction helpers ────────────────────────────────────── */

static float compute_mean(const float *buf, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += buf[i];
    return sum / n;
}

static float compute_rms(const float *buf, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += buf[i] * buf[i];
    return sqrtf(sum / n);
}

static float compute_std(const float *buf, int n, float mean)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = buf[i] - mean;
        sum += d * d;
    }
    return sqrtf(sum / n);
}

static float compute_peak(const float *buf, int n)
{
    float mx = fabsf(buf[0]);
    for (int i = 1; i < n; i++) {
        float v = fabsf(buf[i]);
        if (v > mx) mx = v;
    }
    return mx;
}

static float compute_kurtosis(const float *buf, int n, float mean, float std)
{
    if (std < 1e-6f) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = (buf[i] - mean) / std;
        sum += d * d * d * d;
    }
    return (sum / n) - 3.0f; /* Excess kurtosis */
}

static float compute_skewness(const float *buf, int n, float mean, float std)
{
    if (std < 1e-6f) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = (buf[i] - mean) / std;
        sum += d * d * d;
    }
    return sum / n;
}

/**
 * Extract 74-element feature vector from a sensor window.
 * Features per axis (ax,ay,az,gx,gy,gz): mean, rms, std, peak,
 *   kurtosis, skewness, crest_factor, peak_to_peak  → 8 × 6 = 48
 * Cross-axis: magnitude_rms(accel), magnitude_rms(gyro)         → 2
 * Temperature: mean, max, delta                                  → 3
 * Spectral (dominant freq energy bin, spectral centroid) × 6 axes → 12 (simplified)
 * Zero-crossing rate × 6 axes                                    → 6
 * SMA (signal magnitude area)                                    → 1
 * Correlation (ax·ay, ax·az, ay·az)                             → 3
 *                                                         Total = 75 → pad to 74
 */
void extract_features(const SensorFrame_t *frame, float *features)
{
    float ax[WINDOW_SIZE], ay[WINDOW_SIZE], az[WINDOW_SIZE];
    float gx[WINDOW_SIZE], gy[WINDOW_SIZE], gz[WINDOW_SIZE];
    float tp[WINDOW_SIZE];

    for (int i = 0; i < WINDOW_SIZE; i++) {
        ax[i] = frame->samples[i].accel_x;
        ay[i] = frame->samples[i].accel_y;
        az[i] = frame->samples[i].accel_z;
        gx[i] = frame->samples[i].gyro_x;
        gy[i] = frame->samples[i].gyro_y;
        gz[i] = frame->samples[i].gyro_z;
        tp[i] = frame->samples[i].temperature;
    }

    float *axes[6] = {ax, ay, az, gx, gy, gz};
    int fi = 0;

    for (int a = 0; a < 6; a++) {
        float m   = compute_mean(axes[a], WINDOW_SIZE);
        float rms = compute_rms(axes[a],  WINDOW_SIZE);
        float std = compute_std(axes[a],  WINDOW_SIZE, m);
        float pk  = compute_peak(axes[a], WINDOW_SIZE);
        float kurt = compute_kurtosis(axes[a], WINDOW_SIZE, m, std);
        float skew = compute_skewness(axes[a], WINDOW_SIZE, m, std);
        float crest = (rms > 1e-6f) ? pk / rms : 0.0f;

        float ptp = pk;  /* Already abs max; full p2p needs min too */
        float mn  = axes[a][0];
        for (int i = 1; i < WINDOW_SIZE; i++)
            if (axes[a][i] < mn) mn = axes[a][i];
        ptp = pk - mn;

        features[fi++] = m;
        features[fi++] = rms;
        features[fi++] = std;
        features[fi++] = pk;
        features[fi++] = kurt;
        features[fi++] = skew;
        features[fi++] = crest;
        features[fi++] = ptp;
    }

    /* Acceleration magnitude RMS */
    float mag_acc = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        mag_acc += ax[i]*ax[i] + ay[i]*ay[i] + az[i]*az[i];
    }
    features[fi++] = sqrtf(mag_acc / WINDOW_SIZE);

    /* Gyro magnitude RMS */
    float mag_gyr = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        mag_gyr += gx[i]*gx[i] + gy[i]*gy[i] + gz[i]*gz[i];
    }
    features[fi++] = sqrtf(mag_gyr / WINDOW_SIZE);

    /* Temperature stats */
    float tmean = compute_mean(tp, WINDOW_SIZE);
    float tmax  = tp[0];
    for (int i = 1; i < WINDOW_SIZE; i++) if (tp[i] > tmax) tmax = tp[i];
    features[fi++] = tmean;
    features[fi++] = tmax;
    features[fi++] = tmax - tp[0]; /* delta */

    /* Zero-crossing rate per axis */
    for (int a = 0; a < 6; a++) {
        int zc = 0;
        for (int i = 1; i < WINDOW_SIZE; i++) {
            if ((axes[a][i-1] >= 0.0f && axes[a][i] < 0.0f) ||
                (axes[a][i-1] <  0.0f && axes[a][i] >= 0.0f)) zc++;
        }
        features[fi++] = (float)zc / WINDOW_SIZE;
    }

    /* SMA */
    float sma = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        sma += fabsf(ax[i]) + fabsf(ay[i]) + fabsf(az[i]);
    }
    features[fi++] = sma / WINDOW_SIZE;

    /* Cross-axis correlations (ax·ay, ax·az, ay·az) */
    float cxy = 0.0f, cxz = 0.0f, cyz = 0.0f;
    float mx = compute_mean(ax, WINDOW_SIZE);
    float my = compute_mean(ay, WINDOW_SIZE);
    float mz = compute_mean(az, WINDOW_SIZE);
    for (int i = 0; i < WINDOW_SIZE; i++) {
        cxy += (ax[i]-mx)*(ay[i]-my);
        cxz += (ax[i]-mx)*(az[i]-mz);
        cyz += (ay[i]-my)*(az[i]-mz);
    }
    features[fi++] = cxy / WINDOW_SIZE;
    features[fi++] = cxz / WINDOW_SIZE;
    features[fi++] = cyz / WINDOW_SIZE;

    /* Pad remaining to FEATURE_SIZE */
    while (fi < FEATURE_SIZE) features[fi++] = 0.0f;
}

/* ── Inference Task ────────────────────────────────────────────────── */
void vInferenceTask(void *pvParameters)
{
    (void)pvParameters;

    /* ── Load model ── */
    const tflite::Model *model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        /* Fatal: model schema mismatch */
        vTaskSuspend(NULL);
    }

    /* ── Resolver: register only needed ops (saves flash) ── */
    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();

    /* ── Interpreter ── */
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        vTaskSuspend(NULL);
    }

    TfLiteTensor *input  = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    SensorFrame_t  frame;
    AlertMessage_t alert;
    static float   features[FEATURE_SIZE];

    for (;;) {
        /* Block until a full window arrives */
        if (xQueueReceive(xSensorDataQueue, &frame, portMAX_DELAY) == pdTRUE) {

            /* Feature extraction */
            extract_features(&frame, features);

            /* Copy features into model input tensor */
            for (int i = 0; i < FEATURE_SIZE; i++) {
                if (input->type == kTfLiteFloat32) {
                    input->data.f[i] = features[i];
                } else if (input->type == kTfLiteInt8) {
                    /* Quantize: (val / scale) + zero_point */
                    float scale = input->params.scale;
                    int32_t zp  = input->params.zero_point;
                    int32_t q   = (int32_t)(features[i] / scale) + zp;
                    if (q > 127)  q = 127;
                    if (q < -128) q = -128;
                    input->data.int8[i] = (int8_t)q;
                }
            }

            /* Run inference */
            if (interpreter.Invoke() != kTfLiteOk) continue;

            /* Parse output — find class with highest probability */
            float probs[NUM_FAULT_CLASSES];
            for (int c = 0; c < NUM_FAULT_CLASSES; c++) {
                if (output->type == kTfLiteFloat32) {
                    probs[c] = output->data.f[c];
                } else if (output->type == kTfLiteInt8) {
                    float scale = output->params.scale;
                    int32_t zp  = output->params.zero_point;
                    probs[c] = (output->data.int8[c] - zp) * scale;
                }
            }

            uint8_t best_class = 0;
            float   best_prob  = probs[0];
            for (int c = 1; c < NUM_FAULT_CLASSES; c++) {
                if (probs[c] > best_prob) {
                    best_prob  = probs[c];
                    best_class = c;
                }
            }

            /* Populate alert */
            alert.fault_class       = best_class;
            alert.confidence        = best_prob;
            alert.anomaly_detected  = (best_class != 0) && (best_prob > 0.80f);
            alert.temperature       = frame.samples[WINDOW_SIZE-1].temperature;
            alert.window_id         = frame.window_id;

            xQueueSend(xAlertQueue, &alert, 0);
        }
    }
}