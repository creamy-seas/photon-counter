#include <thread>       // std::thread
#include <mutex>        // std::mutex, std::lock_guard
#include <iostream>     // std::count

#include "colours.hpp"
#include "ia_ADQAPI.hpp"
// https://stackoverflow.com/questions/10828001/pthread-mutex-locking-without-global-mutex
// example of non global locking
int g_i[3] = {0};
std::mutex g_i_mutex;

void safe_increment(int increment)
{
        const std::lock_guard<std::mutex> lock(g_i_mutex);

        g_i[0] = increment;
        std::cout << std::this_thread::get_id() << ": " << g_i[0] << '\n';
        // goes put out scope
}


int main(void){

        master_setup(1);

        // OKBLUE("Running");
        // std::cout << "g_i[0]" << g_i[0] << '\n';

        // std::thread t1(safe_increment, 2);
        // std::thread t2(safe_increment, 4);

        // t1.join();
        // t2.join();

        // std::cout << "g_i[0] " << g_i[0] << '\n';
}
