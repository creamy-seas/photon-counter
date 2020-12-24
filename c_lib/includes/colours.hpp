#ifndef _COLOURS
#define _COLOURS

#define ENDC "\033[0m"
#define BOLD "\033[1m"
#define COLOR(colour, str) (printf("%s%s%s\n", colour, str, ENDC))

#define HEADER(str) (COLOR("\033[95m\033[1m", str))
#define OKBLUE(str) (COLOR("\033[94m\033[1m", str))
#define OKGREEN(str) (COLOR("\033[92m\033[1m", str))
#define WARNING(str) (COLOR("\033[93m\033[1m", str))
#define FAIL(str) (COLOR("\033[91m\033[1m", str))

#define UNDERLINE(str) (COLOR("\033[4m", str))
#define BLINK(str) (COLOR("\033[5m", str))

#define BLACK(str) (COLOR("\033[30m", str))
#define RED(str) (COLOR("\033[31m", str))
#define GREEN(str) (COLOR("\033[32m", str))
#define YELLOW(str) (COLOR("\033[33m", str))
#define BLUE(str) (COLOR("\033[34m", str))
#define VIOLET(str) (COLOR("\033[35m", str))
#define BEIGE(str) (COLOR("\033[91m", str))
#define WHITE(str) (COLOR("\033[37m", str))
#define ROMAN(str) (COLOR("\033[1;34m", str))
#define SKYBLUE(str) (COLOR("\033[1;36m", str))

#define BLACKBG(str) (COLOR("\033[40m", str))
#define REDBG(str) (COLOR("\033[41m", str))
#define GREENBG(str) (COLOR("\033[42m", str))
#define YELLOWBG(str) (COLOR("\033[43m", str))
#define BLUEBG(str) (COLOR("\033[44m", str))
#define VIOLETBG(str) (COLOR("\033[45m", str))
#define BEIGEBG(str) (COLOR("\033[46m", str))
#define WHITEBG(str) (COLOR("\033[47m", str))
#define ROMANBG(str) (COLOR("\033[1;44m", str))
#define SKYBLUEBG(str) (COLOR("\033[1;46m", str))
#endif
