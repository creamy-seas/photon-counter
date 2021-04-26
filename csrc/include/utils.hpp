///////////////////////////////////////////////////////////////////////////////
//                Generic file operation and conversion tools                //
///////////////////////////////////////////////////////////////////////////////
#include <cstring>
#include <fstream>
#include <string>
#include <math.h>
#include <sstream>
#include <string>

#include "colours.hpp"

#ifndef UTILS_HPP
#define UTILS_HPP

std::string seconds_to_hours(int seconds);

std::string float_to_string(float numberToUse, int precision);

template <typename T1=short, typename T2=double> void cast_arrays(
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


template <typename T1> void dump_arrays_to_file(
        T1** array_to_dump,
        int number_of_series,
        int samples_per_record,
        std::string file_name,
        std::string headline_comment
        ) {
        /*
          array_to_dump (x by y):   an array with the format [[a1,a2,a3,...], [b1,b2,b3,...], [c1,c2,c3...],...]
          x=number_of_series:       e.g. 3 for chA, chB, sq
          y=sampler_per_record:			number of elements in each sub array e.g. 400

          written as:
          a1 b1 c1 ... -> chA, chB, sq, etc (x)
          a2 b2 c2 ...
          ...
          ↓
          samples_per_record (y)
        */

        std::fstream fout;
        fout.open(file_name, std::ios_base::out);

        fout << headline_comment;
        for (int i(0); i < samples_per_record; i++) {
                fout << std::endl;
                for (int j(0); j < number_of_series; j++)
                        fout << array_to_dump[j][i] << '\t';
        }
        fout.close();
}


template <typename T> void load_arrays_from_file(
        T** array_to_fill,
        std::string file_name,
        int x_dim,
        int y_dim) {
        /*
        a1 b1 c1 ... -> x-dim
        a2 b2 c2
        ...
        ↓
        y-dim

        Creates arrays:
        [[a1, a2, a3], [b1, b2, b3]]
        */

        std::string line;
        std::stringstream ss;
        int x, y(0);
        T temp_variable;

        std::ifstream fin(file_name);
        if (fin.is_open()) {
                while (!fin.eof()) {

                        // process line
                        getline(fin, line);
                        if ((line[0] != '#') && (line.length() != 0)){
                                ss << line;

                                for (x = 0; x < x_dim; x++) {
                                        ss >> temp_variable;
                                        array_to_fill[x][y] = (T)temp_variable;
                                }
                                //advance to next row
                                y++;
                        }
                        //clear string stream
                        ss.str(std::string());
                        ss.clear();
                }
                if (y > y_dim){
                        FAIL("Array too small in y-direction!");
                }
                if (y < y_dim){
                                FAIL("Array too large in y-direction!");
                                }
        }
        else{
                FAIL("No such file!");
        }
}

#endif
