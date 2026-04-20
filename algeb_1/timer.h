#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
    Timer() : m_start(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        m_start = std::chrono::high_resolution_clock::now();
    }

    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - m_start).count();
    }

private:
    std::chrono::high_resolution_clock::time_point m_start;
};

#endif // TIMER_H