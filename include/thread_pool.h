#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <exception>

namespace diskann
{
class ThreadPool
{
  public:
    ThreadPool(size_t num_threads);

    ~ThreadPool();

    void enqueue_task(std::function<void()> task);

    void wait_for_tasks();

    std::thread::id get_current_thread_id() const;

  private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::condition_variable finish_condition;
    std::atomic<size_t> active_tasks;
    bool stop;
};
} // namespace diskann