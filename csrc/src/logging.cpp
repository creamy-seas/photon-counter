#include <fstream>

#include "logging.hpp"

void append_to_log_file (std::string error_string){
    std::fstream fout;
    fout.open(LOG_FILE, std::ios_base::out|std::ios_base::app);
    fout << error_string + "\n";
    fout.close();
};
