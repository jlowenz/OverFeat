#include "of_util.hpp"

THTensor* THFromData(float* data, int d1, int d2, int d3) {
  if (d2 == -1) {
    THStorage* storage = THStorage_(newWithData)(data, d1);
    return THTensor_(newWithStorage1d)(storage, 0, d1, 1);
  } else if (d3 == -1) {
    THStorage* storage = THStorage_(newWithData)(data, d1*d2);
    return THTensor_(newWithStorage2d)(storage, 0, d1, d2, d2, 1);
    return NULL;
  } else {
    THStorage* storage = THStorage_(newWithData)(data, d1*d2*d3);
    return THTensor_(newWithStorage3d)(storage, 0, d1, d2*d3, d2, d3, d3, 1);
  }
}

THTensor* THFromContiguousArray(PyArrayObject* array) {
  int ndim = PyArray_NDIM(array);
  assert(PyArray_TYPE(array) = FLOAT32_DTYPE);
  assert(ndim < 4);
  npy_intp* dims = PyArray_DIMS(array);
  float* data = (float*)PyArray_DATA(array);
  if (ndim == 1)
    return THFromData(data, dims[0]);
  else if (ndim == 2)
    return THFromData(data, dims[0], dims[1]);
  return THFromData(data, dims[0], dims[1], dims[2]);
}

PyObject* ArrayFromTH(THTensor* th)
{
  npy_intp sizes[3] = {0,0,0};
  for (int i = 0; i < th->nDimension; ++i) {
    sizes[i] = th->size[i];
    //printf("size %d: %d\n", i, sizes[i]);
  }

  //printf("ndim %d, data %lu\n", th->nDimension, THTensor_(data)(th));
  PyArrayObject* output = (PyArrayObject*)PyArray_SimpleNewFromData(th->nDimension,
								    sizes,
								    NPY_FLOAT,
								    THTensor_(data)(th));  
  return PyArray_Return(output);
}

