//
// Created by jiashuai on 17-9-14.
//

#ifndef THUNDERSVM_THUNDERSVM_H
#define THUNDERSVM_THUNDERSVM_H
#include "math.h"
#include "util/common.h"
#include "util/log.h"
#include <CL/sycl.hpp>
#include <cstdlib>
#include <string>
#include <thundersvm/config.h>
#include <vector>
using std::string;
using std::vector;
typedef double float_type;

#define USE_ONEAPI

#ifdef USE_DOUBLE
typedef double kernel_type;
#else
typedef float kernel_type;
#endif
#endif //THUNDERSVM_THUNDERSVM_H
