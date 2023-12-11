#pragma once

/**
 * Copyright (c) Alexander Kurtz 2023
 */

#include <cstdint>

/**
 * This is for my convenience and to make code more readable
 */

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

/**
 * Type that stores a unique action to be performed on the board
 * Save memory if 2d -> 1d index mapping fits in 2^8
 */

#if Boardsize < 16
typedef u8 index_t;
#else
typedef u16 index_t;
#endif
