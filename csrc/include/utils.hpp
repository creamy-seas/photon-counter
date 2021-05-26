#include <fstream> // for writting to files
#include <sstream> // for string stream
#include <string> // supporting stream stream

#include "colours.hpp"

#ifndef UTILS_HPP
#define UTILS_HPP

// Macros from printing hard-coded parameters
#define xstr(s) _str(s)
#define _str(s) #s

/**
 * Convert seconds to human readable format
 */
std::string seconds_to_hours(int seconds);

/**
 * Convert float to a string with a set precisiion value
 */
std::string float_to_string(float numberToUse, int precision);

/**
 * Takes the input arrays and casts them to arrays of another type. This is a very slow method - for better performace try:
 * `std::copy(array_in, array_in + array_size, arrays_out);`
 *
 * @tparam T1 input array type
 * @tparam T2 output array type
 * @param number_of_arrays, array_size
*/
template <typename T1=short, typename T2=double>
void cast_arrays(
    T1** arrays_in, T2** arrays_out,
    int number_of_arrays,
    int array_size) {
    WARNING("Very slow!");

    for (int array_no(0); array_no < number_of_arrays; array_no++) {
        for (int i(0); i < array_size; i++)
            arrays_out[array_no][i] = (T2)arrays_in[array_no][i];
    }
}

/**
 * Supplied array is written to file in the following format:
 *    a1 b1 c1 ... -> chA, chB, sq, ... series
 *    a2 b2 c2 ...
 *    ...
 *    ↓
 *    entries_per_series
 *
 * @param array_to_dump An X by Y array of the format `[[a1,a2,a3,...], [b1,b2,b3,...], [c1,c2,c3...],...]`
 *                      where X is the `number_of_series` e.g. 3 for chA, chB, sq
 *                      and Y is the `entries_per_series` (number of elements in each array) e.g. 4000.
 * @param normalisation Every value is normalised by this before dumping to file
 * @tparam T1 input array type
 * @tparam T2 type to cast the array into before writting to file
 */
template <typename T1, typename T2>
void dump_arrays_to_file(
    T1** array_to_dump,
    int number_of_series,
    int entries_per_series,
    std::string file_name,
    std::string headline_comment,
    T2 normalisation
    ) {

    std::fstream fout;
    fout.open(file_name, std::ios_base::out);

    fout << headline_comment;
    for (int i(0); i < entries_per_series; i++) {
        fout << std::endl;
        for (int j(0); j < number_of_series; j++)
            fout << (double)array_to_dump[j][i] / normalisation << '\t';
    }
    fout.close();
}

/**
 * a1 b1 c1 ... -> x-dim
 * a2 b2 c2
 * ...
 * ↓
 * y-dim
 *
 * Creates arrays:
 * [[a1, a2, a3], [b1, b2, b3]]
 */
template <typename T>
void load_arrays_from_file(
    T** array_to_fill,
    std::string file_name,
    int x_dim, int y_dim) {

    std::string line;
    std::stringstream ss;
    int x, y(0);
    T temp_variable;

    std::ifstream fin(file_name);
    if (fin.is_open()) {
        while (!fin.eof()) {

            // Process line
            getline(fin, line);
            if ((line[0] != '#') && (line.length() != 0)){
                ss << line;

                for (x = 0; x < x_dim; x++) {
                    ss >> temp_variable;
                    array_to_fill[x][y] = (T)temp_variable;
                }
                //Advance to next row
                y++;
            }
            // Clear string stream
            ss.str(std::string());
            ss.clear();
        }
        if (y > y_dim) FAIL("Array too small in y-direction!");
        if (y < y_dim) FAIL("Array too large in y-direction!");
    }
    else FAIL("No such file!");
}

#endif
