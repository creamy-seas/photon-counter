///////////////////////////////////////////////////////////////////////////////
//                          Shorthand colour macros                          //
///////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdexcept>

#ifndef _COLOURS_
#define _COLOURS_

#define ENDC "\033[0m"
#define BOLD "\033[1m"
#define PRINT_COLOR(colour, str) (printf("%s%s%s\n", colour, str, ENDC))

#define HEADER(str) PRINT_COLOR("\033[95m\033[1m", str)
#define OKBLUE(str) PRINT_COLOR("\033[94m\033[1m", str)
#define OKGREEN(str) PRINT_COLOR("\033[92m\033[1m", str)
#define WARNING(str) PRINT_COLOR("\033[93m\033[1m", str)
#define FAIL(str) {                                  \
                PRINT_COLOR("\033[91m\033[1m", str); \
                throw std::runtime_error(str);       \
        }

#define UNDERLINE(str) (PRINT_COLOR("\033[4m", str))
#define FLASH(str) (PRINT_COLOR("\033[5m", str))

#define BLACK(str) (PRINT_COLOR("\033[30m", str))
#define RED(str) (PRINT_COLOR("\033[31m", str))
#define GREEN(str) (PRINT_COLOR("\033[32m", str))
#define YELLOW(str) (PRINT_COLOR("\033[33m", str))
#define BLUE(str) (PRINT_COLOR("\033[34m", str))
#define VIOLET(str) (PRINT_COLOR("\033[35m", str))
#define BEIGE(str) (PRINT_COLOR("\033[91m", str))
#define WHITE(str) (PRINT_COLOR("\033[37m", str))
#define ROMAN(str) (PRINT_COLOR("\033[1;34m", str))
#define SKYBLUE(str) (PRINT_COLOR("\033[1;36m", str))

#define BLACKBG(str) (PRINT_COLOR("\033[40m", str))
#define REDBG(str) (PRINT_COLOR("\033[41m", str))
#define GREENBG(str) (PRINT_COLOR("\033[30m\033[42m", str))
#define YELLOWBG(str) (PRINT_COLOR("\033[30m\033[43m", str))
#define BLUEBG(str) (PRINT_COLOR("\033[44m", str))
#define VIOLETBG(str) (PRINT_COLOR("\033[45m", str))
#define BEIGEBG(str) (PRINT_COLOR("\033[46m", str))
#define WHITEBG(str) (PRINT_COLOR("\033[47m", str))
#define ROMANBG(str) (PRINT_COLOR("\033[1;44m", str))
#define SKYBLUEBG(str) (PRINT_COLOR("\033[1;46m", str))

#endif
