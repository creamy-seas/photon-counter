#include <string> // for std::string
#include <math.h> // floor and ceil
#include <iomanip> // std::setprecision
#include <sstream> // for ss

#include "utils.hpp"

std::string seconds_to_hours(int seconds) {
    int hours = floor(seconds / 3600);
    int minutes = ceil((seconds - hours * 3600) / 60);

    return std::to_string(hours) + "h:" + std::to_string(minutes) + "m";
}

std::string float_to_string(float numberToUse, int precision){
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << numberToUse;
    return ss.str();
}
