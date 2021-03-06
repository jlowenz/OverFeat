THTensor* shared_weights::fprop1(THTensor* input, THTensor** outputs, int net_idx, int max_layer) {
  if (net_idx == 0) {
    Normalization_updateOutput(input, 118.380948, 61.896913, outputs[0]);
    SpatialConvolution_updateOutput(outputs[0], 4, 4, weights[1], bias[1], outputs[1]);
    Threshold_updateOutput(outputs[1], 0.000000, 0.000001, outputs[2]);
    SpatialMaxPooling_updateOutput(outputs[2],2,2,2,2,weights[3], outputs[3]);
    SpatialConvolution_updateOutput(outputs[3], 1, 1, weights[4], bias[4], outputs[4]);
    Threshold_updateOutput(outputs[4], 0.000000, 0.000001, outputs[5]);
    SpatialMaxPooling_updateOutput(outputs[5],2,2,2,2,weights[6], outputs[6]);
    SpatialZeroPadding_updateOutput(outputs[6], 1, 1, 1, 1, outputs[7]);
    SpatialConvolution_updateOutput(outputs[7], 1, 1, weights[8], bias[8], outputs[8]);
    Threshold_updateOutput(outputs[8], 0.000000, 0.000001, outputs[9]);
    SpatialZeroPadding_updateOutput(outputs[9], 1, 1, 1, 1, outputs[10]);
    SpatialConvolution_updateOutput(outputs[10], 1, 1, weights[11], bias[11], outputs[11]);
    Threshold_updateOutput(outputs[11], 0.000000, 0.000001, outputs[12]);
    SpatialZeroPadding_updateOutput(outputs[12], 1, 1, 1, 1, outputs[13]);
    SpatialConvolution_updateOutput(outputs[13], 1, 1, weights[14], bias[14], outputs[14]);
    Threshold_updateOutput(outputs[14], 0.000000, 0.000001, outputs[15]);
    SpatialMaxPooling_updateOutput(outputs[15],2,2,2,2,weights[16], outputs[16]);
    SpatialConvolution_updateOutput(outputs[16], 1, 1, weights[17], bias[17], outputs[17]);
    Threshold_updateOutput(outputs[17], 0.000000, 0.000001, outputs[18]);
    SpatialConvolution_updateOutput(outputs[18], 1, 1, weights[19], bias[19], outputs[19]);
    Threshold_updateOutput(outputs[19], 0.000000, 0.000001, outputs[20]);
    SpatialConvolution_updateOutput(outputs[20], 1, 1, weights[21], bias[21], outputs[21]);
    return outputs[21];
  }
  // TODO: do the max_layer thing for the net_idx == 0
  if (net_idx == 1) {
    Normalization_updateOutput(input, 118.380948, 61.896913, outputs[0]);
    SpatialConvolution_updateOutput(outputs[0], 2, 2, weights[1], bias[1], outputs[1]);
    Threshold_updateOutput(outputs[1], 0.000000, 0.000001, outputs[2]);
    SpatialMaxPooling_updateOutput(outputs[2],3,3,3,3,weights[3], outputs[3]);
    if (max_layer == 3) return outputs[3];
    SpatialConvolution_updateOutput(outputs[3], 1, 1, weights[4], bias[4], outputs[4]);
    if (max_layer == 4) return outputs[4];
    Threshold_updateOutput(outputs[4], 0.000000, 0.000001, outputs[5]);
    if (max_layer == 5) return outputs[5];
    SpatialMaxPooling_updateOutput(outputs[5],2,2,2,2,weights[6], outputs[6]);
    if (max_layer == 6) return outputs[6];
    SpatialZeroPadding_updateOutput(outputs[6], 1, 1, 1, 1, outputs[7]);
    if (max_layer == 7) return outputs[7];
    SpatialConvolution_updateOutput(outputs[7], 1, 1, weights[8], bias[8], outputs[8]);
    if (max_layer == 8) return outputs[8];
    Threshold_updateOutput(outputs[8], 0.000000, 0.000001, outputs[9]);
    if (max_layer == 9) return outputs[9];
    SpatialZeroPadding_updateOutput(outputs[9], 1, 1, 1, 1, outputs[10]);
    if (max_layer == 10) return outputs[10];
    SpatialConvolution_updateOutput(outputs[10], 1, 1, weights[11], bias[11], outputs[11]);
    if (max_layer == 11) return outputs[11];
    Threshold_updateOutput(outputs[11], 0.000000, 0.000001, outputs[12]);
    if (max_layer == 12) return outputs[12];
    SpatialZeroPadding_updateOutput(outputs[12], 1, 1, 1, 1, outputs[13]);
    if (max_layer == 13) return outputs[13];
    SpatialConvolution_updateOutput(outputs[13], 1, 1, weights[14], bias[14], outputs[14]);
    if (max_layer == 14) return outputs[14];
    Threshold_updateOutput(outputs[14], 0.000000, 0.000001, outputs[15]);
    if (max_layer == 15) return outputs[15];
    SpatialZeroPadding_updateOutput(outputs[15], 1, 1, 1, 1, outputs[16]);
    if (max_layer == 16) return outputs[16];
    SpatialConvolution_updateOutput(outputs[16], 1, 1, weights[17], bias[17], outputs[17]);
    if (max_layer == 17) return outputs[17];
    Threshold_updateOutput(outputs[17], 0.000000, 0.000001, outputs[18]);
    if (max_layer == 18) return outputs[18];
    SpatialMaxPooling_updateOutput(outputs[18],3,3,3,3,weights[19], outputs[19]);
    if (max_layer == 19) return outputs[19];
    SpatialConvolution_updateOutput(outputs[19], 1, 1, weights[20], bias[20], outputs[20]);
    if (max_layer == 20) return outputs[20];
    Threshold_updateOutput(outputs[20], 0.000000, 0.000001, outputs[21]);
    if (max_layer == 21) return outputs[21];
    SpatialConvolution_updateOutput(outputs[21], 1, 1, weights[22], bias[22], outputs[22]);
    if (max_layer == 22) return outputs[22];
    Threshold_updateOutput(outputs[22], 0.000000, 0.000001, outputs[23]);
    if (max_layer == 23) return outputs[23];
    SpatialConvolution_updateOutput(outputs[23], 1, 1, weights[24], bias[24], outputs[24]);
    return outputs[24];
  }
}
