#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include "weighted_average_wirelength/src/functional.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
DREAMPLACE_BEGIN_NAMESPACE

/// @brief Compute weighted average wirelength and gradient.
/// WL = \sum_i x_i*exp(x_i/gamma) / \sum_i exp(x_i/gamma) - \sum_i
/// x_i*exp(-x_i/gamma) / \sum_i x_i*exp(-x_i/gamma), where x_i is pin location.
///
/// @param x x location of pins.
/// @param y y location of pins.
/// @param flat_netpin consists pins of each net, pins belonging to the same net
/// are abutting to each other.
/// @param netpin_start bookmark for the starting index of each net in
/// flat_netpin. The length is number of nets. The last entry equals to the
/// number of pins.
/// @param net_mask whether compute the wirelength for a net or not
/// @param net_weights weight of nets
/// @param num_nets number of nets.
/// @param inv_gamma the inverse number of gamma coefficient in weighted average
/// wirelength.
/// @param partial_wl wirelength in x and y directions of each net. The first
/// half is the wirelength in x direction, and the second half is the wirelength
/// in y direction.
/// @param grad_tensor back-propagated gradient from previous stage.
/// @param grad_x_tensor gradient in x direction.
/// @param grad_y_tensor gradient in y direction.
/// @return 0 if successfully done.
template <typename T>


int computeWeightedAverageWirelengthByPinLauncher(
  const T *x, const T *y, const int *pin2net_map, const int *flat_netpin, const int *netpin_start,
    const unsigned char *net_mask, int num_nets, int num_pins, const T *inv_gamma,
    T *partial_wl, T *grad_intermediate_x, T *grad_intermediate_y,
    //T *sum_exp_x, T *sum_exp_nx, T *sum_exp_y, T *sum_exp_ny,
    //T *b_x, T *b_y, T *a_x, T *a_y, T *b_nx, T *b_ny, T *a_nx, T *a_ny, T *xmax, T *xmin, T *ymin, T *ymax,
    int num_threads){


      //using difference_type = std::ptrdiff_t;
      using value_type = T;
      using pointer = T*;
      using reference = T&;
      using iterator_category = std::output_iterator_tag;
      double batch_ratio = 0.5;
      int batch_size = num_pins * batch_ratio;
      //T partialwl[batch_size];
      double gradientbypin[batch_size];
      int arr[num_pins];
      for (int m = 0; m<= num_pins; m++){
        arr[m] = m;
      }
      srand(time(NULL));
      std::random_shuffle(arr,arr+num_pins);
      int newarr[batch_size];
      std::copy(arr, arr+batch_size, newarr);
      double balance = num_pins/batch_size;

      int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_pins/num_threads/32),1);

      //const int num_nets = num_nets;
// ++ it might be int sumexpx, etc
    std::vector<T> sum_exp_x;
    std::vector<T> sum_exp_nx;
    std::vector<T> sum_exp_y;
    std::vector<T> sum_exp_ny;
    std::vector<T> b_x;
    std::vector<T> b_y;
    std::vector<T> a_x;
    std::vector<T> a_y;
    std::vector<T> b_nx;
    std::vector<T> b_ny;
    std::vector<T> a_nx;
    std::vector<T> a_ny;
    std::vector<T> xmax;
    std::vector<T> ymax;
    std::vector<T> xmin;
    std::vector<T> ymin;
    //T sumexpx[num_nets] = {};
    //T sumexpnx[num_nets] = {};
    //T sumexpy[num_nets] = {};
    //T sumexpny[num_nets];
    //T b_x[num_nets];
    //T b_y[num_nets];
    //T a_x[num_nets];
    //T a_y[num_nets];
    //T b_nx[num_nets];
    //T b_ny[num_nets];
    //T a_nx[num_nets];
    //T a_ny[num_nets];
    //T xmax[num_nets];
    //T xmin[num_nets];
    //T ymax[num_nets];
    //T ymin[num_nets];
    for (int n = 0; n < num_nets; n++){
      if (net_mask[n]){
        T exp_x_sum = 0;
        T exp_nx_sum = 0;
        T exp_y_sum = 0;
        T exp_ny_sum = 0;
        T x_exp_x_sum = 0;
        T x_exp_nx_sum = 0;
        T y_exp_y_sum = 0;
        T y_exp_ny_sum = 0;
        T x_max = -std::numeric_limits<T>::max();
        T x_min = std::numeric_limits<T>::max();
        T y_max = -std::numeric_limits<T>::max();
        T y_min = std::numeric_limits<T>::max();
        for (int j = netpin_start[n]; j < netpin_start[n + 1]; ++j) {
          T xx = x[flat_netpin[j]];
          x_max = DREAMPLACE_STD_NAMESPACE::max(xx, x_max);
          x_min = DREAMPLACE_STD_NAMESPACE::min(xx, x_min);
          T yy = y[flat_netpin[j]];
          y_max = DREAMPLACE_STD_NAMESPACE::max(yy, y_max);
          y_min = DREAMPLACE_STD_NAMESPACE::min(yy, y_min);
        
          T exp_x = exp((xx - x_max) * (*inv_gamma));
          T exp_nx = exp((x_min - xx) * (*inv_gamma));

          x_exp_x_sum += xx * exp_x;
          x_exp_nx_sum += xx * exp_nx;
          exp_x_sum += exp_x;
          exp_nx_sum += exp_nx;

          T exp_y = exp((yy - y_max) * (*inv_gamma));
          T exp_ny = exp((y_min - yy) * (*inv_gamma));

          //yexp_y_sum += yy * exp_y;
          //yexp_ny_sum += yy * exp_ny;
          exp_y_sum += exp_y;
          exp_ny_sum += exp_ny;
        }
        sum_exp_x.push_back (exp_x_sum);
        sum_exp_nx.push_back (exp_nx_sum);
        sum_exp_y.push_back(exp_y_sum);
        sum_exp_ny.push_back(exp_ny_sum);

        b_x.push_back((*inv_gamma) / (exp_x_sum));
        a_x.push_back((1.0 - b_x[n] * x_exp_x_sum) / exp_x_sum);
        b_nx.push_back(-(*inv_gamma) / (exp_nx_sum));
        a_nx.push_back((1.0 - b_nx[n] * x_exp_nx_sum) / exp_nx_sum);

        b_y.push_back((*inv_gamma) / (exp_y_sum));
        a_y.push_back((1.0 - b_y[n] * y_exp_y_sum) / exp_y_sum);
        b_ny.push_back(-(*inv_gamma) / (exp_ny_sum));
        a_ny.push_back((1.0 - b_ny[n] * y_exp_ny_sum) / exp_ny_sum);
        xmax.push_back(x_max);
        xmin.push_back(x_min);
        ymax.push_back(y_max);
        ymin.push_back(y_min);
      }
    }
  
  for (int j = newarr[0]; j< newarr[batch_size - 1]; ++j){
    int num_available_net = sizeof(pin2net_map[j])/sizeof(int);
    int available_net[num_available_net];
    std::copy(pin2net_map[j], pin2net_map[j]+num_available_net,available_net);
    T sum_m = 0;
    T grad_x = 0;
    T grady = 0;
    for (int  i = 0; i < num_available_net; i++){
      int net_index = available_net[i];
      //T sum = 0;
      //T gradx = 0;
      //T grady = 0;
      if (net_mask[net_index]){
        int num_related_pins = netpin_start[net_index + 1] - netpin_start[net_index] - 1;
        T xx = x[flat_netpin[j]];
        T yy = y[flat_netpin[j]];
        T xexpx = xx * exp(xx * (*inv_gamma)) / sum_exp_x[net_index];
        T xexpnx = xx * exp(-xx * (*inv_gamma)) / sum_exp_nx[net_index];
        T yexpy = yy * exp(yy * (*inv_gamma)) / sum_exp_y[net_index];
        T yexpny = yy * exp(-yy * (*inv_gamma)) / sum_exp_ny[net_index];
          sum_m = sum_m + xexpx - xexpnx + yexpy - yexpny;

        T exp_x = exp((xx - xmax[net_index]) * (*inv_gamma));
        T exp_nx = exp((xmin[net_index] - xx) * (*inv_gamma));
          grad_x = grad_x + ((a_x[net_index] + b_x[net_index] * xx) * exp_x - (a_nx[net_index] + b_nx[net_index] * xx) * exp_nx);

        T exp_y = exp((yy - ymax[net_index]) * (*inv_gamma));
        T exp_ny = exp((ymin[net_index] - yy) * (*inv_gamma));

        grady = grady + ((a_y[net_index] + b_y[net_index] * yy) * exp_y - (a_ny[net_index] + b_ny[net_index] * yy) * exp_ny);
      }
    }
    partial_wl[j] = sum_m;
    grad_intermediate_x[flat_netpin[j]] = grad_x;
    grad_intermediate_y[flat_netpin[j]] = grady;
  }
  //auto wirelengthtotal = balance * wirelengthbypin.sum;
  return 0;
}


std::vector<at::Tensor> weighted_average_wirelength_by_pin_forward(
    at::Tensor pos, at::Tensor flat_netpin, at::Tensor netpin_start,
    at::Tensor pin2net_map, at::Tensor net_weights, at::Tensor net_mask,
    at::Tensor inv_gamma) {
      
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CPU(net_mask);
  CHECK_CONTIGUOUS(net_mask);
  CHECK_FLAT_CPU(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pos.numel() / 2;

  // x, y interleave
  at::Tensor partial_wl = at::zeros({2000}, pos.options());
  // timed with grad_in yet
  at::Tensor grad_intermediate = at::zeros_like(pos);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeWeightedAverageWirelengthByPinLauncher", [&] {
        computeWeightedAverageWirelengthByPinLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
            DREAMPLACE_TENSOR_DATA_PTR(pin2net_map, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
            DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
            DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), num_nets, num_pins,
            DREAMPLACE_TENSOR_DATA_PTR(inv_gamma, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(partial_wl, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t) + num_pins,
            at::get_num_threads());
        if (net_weights.numel()) {
          partial_wl.mul_(net_weights);
        }
      });

  auto wl = partial_wl.sum();
  return {wl, grad_intermediate};
}

/// @brief Compute gradient
/// @param grad_pos input gradient from backward propagation
/// @param pos locations of pins
/// @param flat_netpin similar to the JA array in CSR format, which is flattened
/// from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is
/// the number of pins in each net, the length of IA is number of nets + 1
/// @param net_weights weight of nets
/// @param net_mask an array to record whether compute the where for a net or
/// not
/// @param inv_gamma a scalar tensor for the parameter in the equation
at::Tensor weighted_average_wirelength_by_pin_backward(
    at::Tensor grad_pos, at::Tensor pos, at::Tensor grad_intermediate,
    at::Tensor flat_netpin, at::Tensor netpin_start, at::Tensor pin2net_map,
    at::Tensor net_weights, at::Tensor net_mask, at::Tensor inv_gamma) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CPU(net_mask);
  CHECK_CONTIGUOUS(net_mask);
  CHECK_FLAT_CPU(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);
  CHECK_FLAT_CPU(grad_intermediate);
  CHECK_EVEN(grad_intermediate);
  CHECK_CONTIGUOUS(grad_intermediate);

  at::Tensor grad_out = grad_intermediate.mul_(grad_pos);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeWeightedAverageWirelengthByPinLauncher", [&] {
        if (net_weights.numel()) {
          integrateNetWeightsLauncher<scalar_t>(
              DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
              DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
              DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char),
              DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + pos.numel() / 2,
              netpin_start.numel() - 1, at::get_num_threads());
        }
      });
  return grad_out ;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::weighted_average_wirelength_by_pin_forward,
        "WeightedAverageWirelength_by_pin forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::weighted_average_wirelength_by_pin_backward,
        "WeightedAverageWirelength_by_pin backward");
}