#ifndef GPUPLACE_WEIGHTED_AVERAGE_WIRELENGTH_FUNCTIONAL_H
#define GPUPLACE_WEIGHTED_AVERAGE_WIRELENGTH_FUNCTIONAL_H

#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include <random>
#include <algorithm>
#include <time.h>
DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void integrateNetWeightsLauncher(
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    const T *net_weights,
    T *grad_x_tensor, T *grad_y_tensor,
    int num_nets,
    int num_threads)
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int net_id = 0; net_id < num_nets; ++net_id)
    {
        if (net_mask[net_id])
        {
            T weight = net_weights[net_id];
            for (int j = netpin_start[net_id]; j < netpin_start[net_id + 1]; ++j)
            {
                int pin_id = flat_netpin[j];
                grad_x_tensor[pin_id] *= weight;
                grad_y_tensor[pin_id] *= weight;
            }
        }
    }
}

// V has to be int, or long long int
template <typename T, typename V>
void computeMaxMinNetByNet(
    const T *x, const T *y,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    V *x_max_ptr,
    V *x_min_ptr,
    int num_threads)
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < num_nets; ++i)
    {
        if (net_mask[i])
        {
            const int x_index = i;
            const int y_index = i + num_nets;

            V x_max = x_max_ptr[x_index];
            V x_min = x_min_ptr[x_index];
            V y_max = x_max_ptr[y_index];
            V y_min = x_min_ptr[y_index];

            for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j)
            {
                T xx = x[flat_netpin[j]];
                x_max = DREAMPLACE_STD_NAMESPACE::max((V)xx, x_max);
                x_min = DREAMPLACE_STD_NAMESPACE::min((V)xx, x_min);

                T yy = y[flat_netpin[j]];
                y_max = DREAMPLACE_STD_NAMESPACE::max((V)yy, y_max);
                y_min = DREAMPLACE_STD_NAMESPACE::min((V)yy, y_min);
            }

            x_max_ptr[x_index] = x_max;
            x_min_ptr[x_index] = x_min;
            x_max_ptr[y_index] = y_max;
            x_min_ptr[y_index] = y_min;
        }
    }
}

template <typename T, typename V>
void computeABCKernelsPinByPin(
    const T *x, const T *y,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    V *x_max, V *x_min,
    T *exp_x, T *exp_nx,
    T *exp_x_sum, T *exp_nx_sum,
    T *xexp_x_sum, T *xexp_nx_sum,
    int num_threads)
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_pins / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < num_pins; ++i)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            exp_x[i] = exp((x[i] - x_max[net_id]) * (*inv_gamma));
            exp_nx[i] = exp((x_min[net_id] - x[i]) * (*inv_gamma));

#pragma omp atomic
            exp_x_sum[net_id] += exp_x[i];
#pragma omp atomic
            exp_nx_sum[net_id] += exp_nx[i];
#pragma omp atomic
            xexp_x_sum[net_id] += x[i] * exp_x[i];
#pragma omp atomic
            xexp_nx_sum[net_id] += x[i] * exp_nx[i];

            net_id += num_nets;
            int pin_id = i + num_pins;
            exp_x[pin_id] = exp((y[i] - x_max[net_id]) * (*inv_gamma));
            exp_nx[pin_id] = exp((x_min[net_id] - y[i]) * (*inv_gamma));

#pragma omp atomic
            exp_x_sum[net_id] += exp_x[pin_id];
#pragma omp atomic
            exp_nx_sum[net_id] += exp_nx[pin_id];
#pragma omp atomic
            xexp_x_sum[net_id] += y[i] * exp_x[pin_id];
#pragma omp atomic
            xexp_nx_sum[net_id] += y[i] * exp_nx[pin_id];
        }
    }
}

template <typename T>
void computeXExpSumByExpSumXY(
    const T *xexp_x_sum, const T *xexp_nx_sum,
    const T *exp_x_sum, const T *exp_nx_sum,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    T *partial_wl,
    int num_threads)
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < num_nets; ++i)
    {
        if (net_mask[i])
        {
            T wl_x = xexp_x_sum[i] / exp_x_sum[i] - xexp_nx_sum[i] / exp_nx_sum[i];
            int y_index = i + num_nets;
            T wl_y = xexp_x_sum[y_index] / exp_x_sum[y_index] - xexp_nx_sum[y_index] / exp_nx_sum[y_index];

            partial_wl[i] = wl_x + wl_y;
        }
    }
}


template<typename T, typename V>
void computepartialwl(const T *x, const T *y, 
    const T *exp_x_sum, const T *exp_nx_sum, V *x_max, V *x_min,
    const T *exp_x, const T *exp_nx,
    const int *pin2net_map, const unsigned char *net_mask, const int *flat_node2pin_start_map, const int *flat_node2pin_map, int num_nodes, int num_pins,
    //const std::vector<int> pins_array, 
    int num_nets, 
    T *partial_wl, int num_threads){
    //int num_nodes = 200000;
    //int num_nodes = sizeof(flat_node2pin_start_map)/sizeof(flat_node2pin_start_map[1]);
    int batch_size = num_nodes*0.1;
    int node_array[num_nodes];
    //std::vector<int> node_array;
    for(int node_index = 0; node_index < num_nodes; node_index ++){
         node_array[node_index]=node_index;
    }

    srand(time(NULL));
    std::random_shuffle(node_array,node_array+num_nodes);
    int newarr[batch_size];
    std::copy(node_array,node_array+batch_size,newarr);

    std::vector<int> pins_array;

    //for (int ind = 0; ind < batch_size; ind++){
    //    int node_index = newarr[ind];
    for (int node_index : newarr){
        for (int idx = flat_node2pin_start_map[node_index]; idx < flat_node2pin_start_map[node_index+1]; idx++){
            int pin_index = flat_node2pin_map[idx];
            pins_array.push_back(pin_index);
        }
    }
    int num_pins_batch = sizeof(pins_array)/sizeof(pins_array[1]);
    double balance = num_pins / num_pins_batch;
    // pins_array.push_back(1);
    // pins_array.push_back(2);
    //int pins_array[10] = {1,2,3,4,5,6,7,8,9,10};
    //int num_pins = sizeof(pin2net_map)/sizeof(int);
//    int num_pin_batch = sizeof(pins_array)/sizeof(int);
//    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_pins_batch/num_threads/16),1);
//#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
//    for (int pinidx = 0; pinidx < num_pins_batch; pinidx ++ ){
//        int pinindex = pins_array[pinidx];
    for (int i : pins_array){
        int parent_net = pin2net_map[i];
        if (net_mask[parent_net]){
            T wl_x = (x[i]*exp_x[i]/exp_x_sum[parent_net] - x[i]*exp_nx[i]/exp_nx_sum[parent_net]);
            int y_index = parent_net + num_nets;
            T wl_y = (y[i]*exp_x[i+num_pins]/exp_x_sum[y_index])-y[i]*exp_nx[i+num_pins]/exp_nx_sum[y_index];
            partial_wl[i] = wl_x+wl_y;
            //partial_wl[i]=1;
        }
//
    }
}
//    for (int i : pins_array){
//        partial_wl[i] = 1;}
//}

template <typename T>
void computeWeightedAverageWirelengthGradPinByPin(
    const T *x, const T *y,
    const T *exp_x, const T *exp_nx,
    const T *exp_x_sum, const T *exp_nx_sum,
    const T *xexp_x_sum, const T *xexp_nx_sum,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    const T *grad_tensor,
    T *grad_x_tensor, T *grad_y_tensor,
    int num_threads)
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_pins / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < num_pins; ++i)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            grad_x_tensor[i] = (*grad_tensor) *
                               (((1 + (*inv_gamma) * x[i]) * exp_x_sum[net_id] - (*inv_gamma) * xexp_x_sum[net_id]) / (exp_x_sum[net_id] * exp_x_sum[net_id]) * exp_x[i] - ((1 - (*inv_gamma) * x[i]) * exp_nx_sum[net_id] + (*inv_gamma) * xexp_nx_sum[net_id]) / (exp_nx_sum[net_id] * exp_nx_sum[net_id]) * exp_nx[i]);

            net_id += num_nets;
            int pin_id = i + num_pins;
            grad_y_tensor[i] = (*grad_tensor) *
                               (((1 + (*inv_gamma) * y[i]) * exp_x_sum[net_id] - (*inv_gamma) * xexp_x_sum[net_id]) / (exp_x_sum[net_id] * exp_x_sum[net_id]) * exp_x[pin_id] - ((1 - (*inv_gamma) * y[i]) * exp_nx_sum[net_id] + (*inv_gamma) * xexp_nx_sum[net_id]) / (exp_nx_sum[net_id] * exp_nx_sum[net_id]) * exp_nx[pin_id]);
        }
    }
}

DREAMPLACE_END_NAMESPACE

#endif
