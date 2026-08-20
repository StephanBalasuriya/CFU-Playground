/*
 * Copyright 2026 The CFU-Playground Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "models/mnv3/mnv3.h"

#include <stdio.h>

#include "menu.h"
#include "models/mnv3/input_00001.h"
#include "models/mnv3/model_mobilenetv3_small_min.h"
#include "tflite.h"

#ifdef CSR_VIDEO_FRAMEBUFFER_BASE
extern "C" {
#include "fb_util.h"
};
#endif

// Initialize the model in TFLM runtime
static void mnv3_init(void) {
  tflite_load_model(model_mobilenetv3_small_min, model_mobilenetv3_small_min_len);
}

// Run MobileNetV3 Minimalistic classification
static int32_t mnv3_classify(void) {
  printf("Running MobileNetV3 Minimalistic model...\n");
  tflite_classify();

  int8_t* output = tflite_get_output();
  return (int32_t)output[1] - (int32_t)output[0];
}

static void do_classify_zeros(void) {
  tflite_set_input_zeros();
  int32_t result = mnv3_classify();
  printf("Result for Zeros Input: %ld\n", (long)result);
}

static void do_classify_test0(void) {
  tflite_set_input_unsigned(input_00001);
  int32_t result = mnv3_classify();
  printf("Result for Test Input 0: %ld\n", (long)result);

#ifdef CSR_VIDEO_FRAMEBUFFER_BASE
  char msg_buff[256] = {0};
  snprintf(msg_buff, sizeof(msg_buff), "Result: %ld", (long)result);
  fb_clear();
  fb_draw_string(0, 10, 0x007FFF00, "MobileNetV3 Min Test 0");
  fb_draw_buffer(0, 50, 160, 160, (const uint8_t*)input_00001, 3);
  fb_draw_string(0, 220, 0x007FFF00, msg_buff);
  flush_cpu_dcache();
  flush_l2_cache();
#endif
}

static struct Menu MENU = {
    "MobileNetV3 Minimalistic Models",
    "mnv3",
    {
        MENU_ITEM('g', "Run golden test input 0", do_classify_test0),
        MENU_ITEM('z', "Run zeros input test", do_classify_zeros),
        MENU_END,
    },
};

void mnv3_menu(void) {
  mnv3_init();
  menu_run(&MENU);
}
