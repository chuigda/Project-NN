#include "mnist.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef union {
    uint16_t v16;
    uint8_t v8;
} nn_endian_probe_t;

static const nn_endian_probe_t g_endian_probe = { .v16 = 0xFEFF };

#define IS_LITTLE_ENDIAN (g_endian_probe.v8 == 0xFF)

static uint32_t u32_from_big_endian(uint8_t *buf) {
    uint32_t ret;
    uint8_t *retbuf = (uint8_t*)&ret;
    if (IS_LITTLE_ENDIAN) {
        for (size_t i = 0; i < 3; i++) {
            retbuf[i] = buf[3 - i];
        }
    } else {
        memcpy(retbuf, buf, 4);
    }
    return ret;
}

#define READ_DWORD(var_name) \
    if (fread(readbuf, 1, 4, fp) != 1) { \
        nn_error_t err; \
        if (feof(fp)) { \
            err = NN_BAD_FORMAT; \
        } else { \
            err = NN_IO_ERROR; \
        } \
        fclose(fp); \
        return err; \
    } \
    uint32_t var_name = u32_from_big_endian(readbuf); \

nn_error_t nn_mnist_load_images(const char *file_name,
                                uint8_t **p_images,
                                size_t *p_cnt,
                                size_t *p_w,
                                size_t *p_h)
{
    FILE *fp = fopen(file_name, "rb");
    if (!fp) {
        return NN_IO_ERROR;
    }

    uint8_t readbuf[4];
    READ_DWORD(magic);
    if (magic != 0x0803) {
        fclose(fp);
        return NN_BAD_FORMAT;
    }

    READ_DWORD(cnt);
    READ_DWORD(h);
    READ_DWORD(w);

    size_t image_size = w * h;
    uint8_t *images = malloc(image_size * cnt);
    if (!images) {
        fclose(fp);
        return NN_OUT_OF_MEMORY;
    }

    if (fread(images, image_size, cnt, fp) != cnt) {
        nn_error_t err;
        if (feof(fp)) {
            err = NN_BAD_FORMAT;
        } else {
            err = NN_IO_ERROR;
        }
        fclose(fp);
        free(images);
        return err;
    }

    *p_images = images;
    *p_cnt = cnt;
    *p_h = h;
    *p_w = w;
    fclose(fp);
    return NN_NO_ERROR;
}



nn_error_t nn_mnist_load_labels(char const *file_name,
                                uint8_t **p_labels,
                                size_t *p_cnt)
{
    FILE *fp = fopen(file_name, "rb");
    if (!fp) {
        return NN_IO_ERROR;
    }

    uint8_t readbuf[4];
    READ_DWORD(magic);
    if (magic != 0x0801) {
        fclose(fp);
        return NN_BAD_FORMAT;
    }

    READ_DWORD(cnt);
    uint8_t *labels = malloc(cnt);
    if (!labels) {
        fclose(fp);
        return NN_OUT_OF_MEMORY;
    }

    if (fread(labels, 1, cnt, fp) != cnt) {
        nn_error_t err;
        if (feof(fp)) {
            err = NN_BAD_FORMAT;
        } else {
            err = NN_IO_ERROR;
        }
        fclose(fp);
        free(labels);
        return err;
    }

    *p_labels = labels;
    *p_cnt = cnt;
    fclose(fp);
    return NN_NO_ERROR;
}
