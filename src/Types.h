#pragma once

/**
 * Copyright (c) Alexander Kurtz 2023
 */

#include <cstdint>

// So it correctly works with BoardSize
#include "Config.h"

/**
 * This is for my convenience and to make code more readable
 */

typedef uint8_t     u8_t;
typedef uint16_t    u16_t;
typedef uint32_t    u32_t;
typedef uint64_t    u64_t;
typedef int8_t      i8_t;
typedef int16_t     i16_t;
typedef int32_t     i32_t;
typedef int64_t     i64_t;

/**
 * Type that stores a unique action to be performed on the board
 * Save memory if 2d -> 1d index mapping fits in 2^8
 */

#if BoardSize < 16
typedef u8_t        index_t;
#else
typedef u16_t       index_t;
#endif
