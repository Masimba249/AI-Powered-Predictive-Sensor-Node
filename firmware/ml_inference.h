#ifndef ML_INFERENCE_H
#define ML_INFERENCE_H

#include "sensor_task.h"
#include "FreeRTOS.h"
#include "task.h"

/* TFLite tensor arena size (bytes) — tune for your model */
#define TENSOR_ARENA_SIZE (32 * 1024)

/**
 * @brief FreeRTOS inference task.
 *        Receives SensorFrame from queue, extracts features,
 *        runs TFLite Micro model, sends AlertMessage if anomaly.
 */
void vInferenceTask(void *pvParameters);

/**
 * @brief Feature vector size fed into the model.
 *        12 statistical features × 6 axes + 2 thermal = 74
 */
#define FEATURE_SIZE 74

/**
 * @brief Extract statistical features from a sensor window.
 * @param frame   Input sensor frame (WINDOW_SIZE samples)
 * @param features Output feature array [FEATURE_SIZE]
 */
void extract_features(const SensorFrame_t *frame, float *features);

#endif /* ML_INFERENCE_H */