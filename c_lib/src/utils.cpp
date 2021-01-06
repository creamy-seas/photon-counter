#include "colours.h"
#include "utils.hpp"

template <typename T1, typename T2> void cast_arrays(
        T1** arrays_in, T2** arrays_out,
        int number_of_arrays,
        int array_size) {
        /*
        Takes the input arrays and casts them to arrays of another type

        The faster method is
        std::copy(array_in, array_in + array_size, arrays_out);
        */
        WARNING("Very slow!");

        for (int array_no(0); array_no < number_of_arrays; array_no++) {
                for (int i(0); i < array_size; i++)
                        arrays_out[array_no][i] = (T2)arrays_in[array_no][i];
        }
}
