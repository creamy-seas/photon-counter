/**
 * Shorthand colour macros
 * Read https://gcc.gnu.org/onlinedocs/cpp/Variadic-Macros.html for explanation
 * of ## and macro argument expansion
 * Read https://www.geeksforgeeks.org/multiline-macros-in-c/ about multiline macros
 */
#include <stdio.h>
#include <stdexcept>
#include <iostream>
#include <string> // for std::string
#include <sstream>

#ifndef _COLOURS_
#define _COLOURS_

/**
 * logging error to a file when library is built for python
 */
void append_to_log_file(std::string);

#ifndef DEBUG
#define DEBUG 0
#endif

#define ENDC "\033[0m"
#define BOLD "\033[1m"
#define PRINT_COLOR(colour, fstr, ...) ({       \
    if (DEBUG) {                                 \
    printf(colour);                              \
    printf(fstr, ##__VA_ARGS__);                 \
    printf("%s\n", ENDC);                        \
    }})

#define HEADER(str, ...) PRINT_COLOR("\033[95m\033[1m", str, ##__VA_ARGS__)
#define OKBLUE(str, ...) PRINT_COLOR("\033[94m\033[1m", str, ##__VA_ARGS__)
#define OKGREEN(str, ...) PRINT_COLOR("\033[92m\033[1m", str, ##__VA_ARGS__)
#define WARNING(str, ...) PRINT_COLOR("\033[93m\033[1m", str, ##__VA_ARGS__)

// For Python logging will be done to a file, since python cannot handle exceptions
#ifdef PYTHON
#define PYTHON_START try {
#define PYTHON_END } catch(...) { return 1;}
#else
#define PYTHON_START
#define PYTHON_END
#endif
#define LOG_FILE "libia.log"
#define FAIL(error_string) {                                            \
        std::ostringstream os ;                                         \
        os << __FILE__ << "(" << __LINE__ << "): " << error_string;     \
                                                                        \
        std::string appended_string = os.str();                         \
        append_to_log_file(appended_string);                            \
        std::cout << "\033[91m\033[1m" << appended_string << ENDC << "\n"; \
        throw std::runtime_error(appended_string);                      \
    }
// Successful result is always 0 for CUDA functions. Anything else is a specific error
#define CUDA_CHECK(result, error_string) {                              \
    if (result) {                                                       \
    std::ostringstream os ;                                             \
    os << error_string;                                                 \
    os << "\n- " << #result;                                            \
    os << "\n- " << "CUDA error " << result << ": " << _cudaGetErrorEnum(result); \
    std::string error = os.str();                                       \
    FAIL(error);                                                        \
    }                                                                   \
    }

#define UNDERLINE(str, ...) PRINT_COLOR("\033[4m", str, ##__VA_ARGS__)
#define FLASH(str, ...) PRINT_COLOR("\033[5m", str, ##__VA_ARGS__)

#define BLACK(str, ...) PRINT_COLOR("\033[30m", str, ##__VA_ARGS__)
#define RED(str, ...) PRINT_COLOR("\033[31m", str, ##__VA_ARGS__)
#define GREEN(str, ...) PRINT_COLOR("\033[32m", str, ##__VA_ARGS__)
#define YELLOW(str, ...) PRINT_COLOR("\033[33m", str, ##__VA_ARGS__)
#define BLUE(str, ...) PRINT_COLOR("\033[34m", str, ##__VA_ARGS__)
#define VIOLET(str, ...) PRINT_COLOR("\033[35m", str, ##__VA_ARGS__)
#define BEIGE(str, ...) PRINT_COLOR("\033[91m", str, ##__VA_ARGS__)
#define WHITE(str, ...) PRINT_COLOR("\033[37m", str, ##__VA_ARGS__)
#define ROMAN(str, ...) PRINT_COLOR("\033[1;34m", str, ##__VA_ARGS__)
#define SKYBLUE(str, ...) PRINT_COLOR("\033[1;36m", str, ##__VA_ARGS__)

#define BLACKBG(str, ...) PRINT_COLOR("\033[40m", str, ##__VA_ARGS__)
#define REDBG(str, ...) PRINT_COLOR("\033[41m", str, ##__VA_ARGS__)
#define GREENBG(str, ...) PRINT_COLOR("\033[30m\033[42m", str, ##__VA_ARGS__)
#define YELLOWBG(str, ...) PRINT_COLOR("\033[30m\033[43m", str, ##__VA_ARGS__)
#define BLUEBG(str, ...) PRINT_COLOR("\033[44m", str, ##__VA_ARGS__)
#define VIOLETBG(str, ...) PRINT_COLOR("\033[45m", str, ##__VA_ARGS__)
#define BEIGEBG(str, ...) PRINT_COLOR("\033[46m", str, ##__VA_ARGS__)
#define WHITEBG(str, ...) PRINT_COLOR("\033[47m", str, ##__VA_ARGS__)
#define ROMANBG(str, ...) PRINT_COLOR("\033[1;44m", str, ##__VA_ARGS__)
#define SKYBLUEBG(str, ...) PRINT_COLOR("\033[1;46m", str, ##__VA_ARGS__)

#endif
