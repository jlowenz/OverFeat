#ifndef __OVERFEAT_OVERFEAT_HPP__
#define __OVERFEAT_OVERFEAT_HPP__

#include "THTensor.hpp"
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include <thread>
#include <mutex>
#include <map>

namespace overfeat {

  struct shared_weights
  {
    shared_weights(const char* filename, int net_idx, int nModules);
    ~shared_weights();

    THTensor* weights[25];
    THTensor* bias[25];   

    THTensor* fprop1(THTensor* input, THTensor** outputs, int net_idx, int max_layer);
    THTensor* load_tensor(int istart, int d1, int d2=-1, int d3=-1, int d4=-1);

    int nModules;
    size_t weight_file_pos;
    FILE* weight_file;
  };

  typedef std::shared_ptr<shared_weights> shared_weight_ptr;

  class Overfeat {
  private:
    std::string weight_file_path_g;
    int net_idx_g;
    int max_layer_g;
    THTensor* outputs[25];
    shared_weight_ptr weights;

    int nModules(int net_idx);
  

  public:

    // Call this function once before any other call to overfeat
    // weight_file_path must be set to the path to the weight file.
    // The default weight file is located at data/default/net_weight
    Overfeat(const std::string & weight_file_path, int net_idx, int max_layer=-1);
    virtual ~Overfeat();
  
  
    // This function computes the feature extraction
    //  input should be a 3xHxW THTensor*
    //  the function returns a nClasses x h x w THTensor*
    //  see README for more details
    // The output tensor must NOT be freed by the user
    THTensor* fprop(THTensor* input);
  
    // This function computes the soft max, transforming the output of the network
    //  input probabilities. See README for more details
    void soft_max(THTensor* input, THTensor* output);
  
    // Returns the number of layers of the network
    int get_n_layers();
  
    // Returns the output of the i-th layer
    THTensor* get_output(int i);
  
    // Returns a string corresponding of the name of the i-th class
    std::string get_class_name(int i);
  
    // Returns a vector of pairs (name, probability), corresponding to the
    //  n most likely classes, in decreasing order.
    // The tensor probas should correspond to probabilities
    //  (ie. it should have been through soft_max), otherwise the probabilities
    //  will be wrong (alghouth the ranking whould be ok)
    std::vector<std::pair<std::string, float> >
    get_top_classes(THTensor* probas, int n);
  };
}


#endif
