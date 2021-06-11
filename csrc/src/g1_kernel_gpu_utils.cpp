#include "g1_kernel.hpp"

int G1::GPU::DIRECT::fetch_g1_kernel_blocks(){
    return 1;
}

int G1::GPU::DIRECT::fetch_g1_kernel_threads(){
    return 1;
}

int G1::GPU::DIRECT::check_g1_kernel_parameters(bool display){
    return 1;
}
