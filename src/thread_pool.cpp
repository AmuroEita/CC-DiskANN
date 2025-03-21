#include "thread_pool.h"

namespace diskann
{
ThreadPool::ThreadPool(size_t numThreads) : stop(false), active_tasks(0)
{
    for (size_t i = 0; i < numThreads; ++i)
    {
        workers.emplace_back([this] {
            while (true)
            {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                    if (this->stop && this->tasks.empty())
                        return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                    ++active_tasks;
                }
                task();
                --active_tasks;
                finish_condition.notify_one();
            }
        });
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers)
        worker.join();
}

void ThreadPool::enqueue_task(std::function<void()> task)
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop)
            throw std::runtime_error("ThreadPool has been stopped.");
        tasks.push(std::move(task));
    }
    condition.notify_one();
}

void ThreadPool::wait_for_tasks()
{
    std::unique_lock<std::mutex> lock(queue_mutex);
    finish_condition.wait(lock, [this] { return tasks.empty() && activeTasks == 0; });
}

std::thread::id ThreadPool::get_current_thread_id() const
{
    return std::this_thread::get_id();
}

} // namespace diskann