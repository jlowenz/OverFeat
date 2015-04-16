#pragma once

#include <Python.h>
#include <cassert>
#include "overfeat.hpp"
#include <numpy/arrayobject.h>

THTensor* THFromData(float* data, int d1, int d2 = -1, int d3 = -1);
THTensor* THFromContiguousArray(PyArrayObject* array);
PyObject* ArrayFromTH(THTensor* th);
