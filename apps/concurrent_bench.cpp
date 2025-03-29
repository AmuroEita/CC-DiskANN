// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <cstring>
#include <boost/program_options.hpp>
#include <atomic>
#include <filesystem>

#include "index.h"
#include "utils.h"
#include "timer.h"
#include "thread_pool.h"
#include "program_options_utils.hpp"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"
#include "ann_exception.h"
#include "index_factory.h"

namespace po = boost::program_options;

struct Event {
    bool is_insert;         
    std::vector<float> data; 
    size_t id;               
    size_t query_idx;        
};

template <typename TagT = uint32_t> struct SearchResult
{
    size_t offset;
    size_t query_idx;
    std::vector<TagT> query_result_tags;

    SearchResult(size_t offset, size_t idx, std::vector<TagT> tags)
        : offset(offset), query_idx(idx), query_result_tags(std::move(tags))
    {
    }
};

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
bool concurrent_bench(const std::string data_path, const std::string &query_file, const size_t begin_num,
                      const float write_ratio, const size_t batch_size, const uint32_t recall_at, const uint32_t Lb,
                      const uint32_t Ls, const uint32_t R, const float alpha, const uint32_t num_threads,
                      const diskann::Metric metric, const bool use_opq, const bool use_pq_build,
                      const uint32_t build_PQ_bytes, const std::string &truthset_file, const std::vector<uint32_t> &Lvec)
{
    diskann::cout << "Starting concurrent benchmarking with R: " << R << "  Lb: " << Lb << "  alpha: " << alpha
                  << " #threads: " << num_threads << " #ratio: " << write_ratio << ":" << 1 - write_ratio 
                  << " event_rate: 10K/s" << std::endl;

    size_t data_num, data_dim, aligned_dim;
    diskann::get_bin_metadata(data_path, data_num, data_dim);
    aligned_dim = data_dim; 

    auto index_build_params = diskann::IndexWriteParametersBuilder(Lb, R)
                                  .with_alpha(alpha)
                                  .with_saturate_graph(false)
                                  .with_num_threads(num_threads)
                                  .build();

    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(data_dim)
                      .with_max_points(data_num)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .with_data_type(diskann_type_to_name<T>())
                      .with_label_type(diskann_type_to_name<LabelT>())
                      .with_tag_type(diskann_type_to_name<TagT>())
                      .is_dynamic_index(true)
                      .with_index_write_params(index_build_params)
                      .is_enable_tags(true)
                      .is_use_opq(use_opq)
                      .is_pq_dist_build(use_pq_build)
                      .with_num_pq_chunks(build_PQ_bytes)
                      .build();

    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();

    T *data = nullptr;
    diskann::load_aligned_bin<T>(data_path, data, data_num, data_dim, aligned_dim);

    T *query = nullptr;
    size_t query_num, query_dim, query_aligned_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    std::vector<uint32_t> tags(begin_num);
    std::iota(tags.begin(), tags.end(), 1 + static_cast<uint32_t>(0));
    index->build(data, begin_num, tags);

    double event_rate = 5000 * num_threads;
    const double experiment_time = 10.0; 

    struct Event {
        bool is_insert;          
        std::vector<T> data;     
        TagT id;                 
        size_t query_idx;        
    };

    std::queue<Event> event_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> running(true);

    std::mutex insert_latency_mutex, search_latency_mutex, result_mutex;
    std::vector<double> insert_latency_stats, search_latency_stats;
    std::vector<std::pair<size_t, std::vector<TagT>>> search_results;
    std::atomic<size_t> insert_count(0), search_count(0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    auto producer = [&]() {
        size_t id = begin_num + 1; 
        size_t query_idx = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            if (elapsed >= experiment_time) break;

            size_t target_events = static_cast<size_t>(event_rate * elapsed);
            while (insert_count + search_count < target_events) {
                Event event;
                event.data.resize(aligned_dim);

                if (dis(gen) < write_ratio) {
                    size_t data_idx = (id - 1) % data_num; 
                    std::copy(data + data_idx * aligned_dim, data + (data_idx + 1) * aligned_dim, event.data.begin());
                    event.is_insert = true;
                    event.id = static_cast<TagT>(id++);
                } else {
                    event.query_idx = query_idx % query_num;
                    std::copy(query + event.query_idx * query_aligned_dim, 
                              query + (event.query_idx + 1) * query_aligned_dim, 
                              event.data.begin());
                    event.is_insert = false;
                    query_idx++;
                }

                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    event_queue.push(event);
                }
                queue_cv.notify_one();

                std::this_thread::sleep_for(std::chrono::microseconds(1000000 / static_cast<int>(event_rate)));
            }
        }
        running = false;
        queue_cv.notify_all();
    };

    auto worker = [&](int thread_id) {
        while (running || !event_queue.empty()) {
            Event event;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (event_queue.empty()) {
                    if (!running) break;
                    queue_cv.wait(lock);
                    continue;
                }
                event = event_queue.front();
                event_queue.pop();
            }

            if (event.is_insert) {
                auto qs = std::chrono::high_resolution_clock::now();
                int insert_result = index->insert_point(event.data.data(), event.id);
                auto qe = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = qe - qs;
                if (insert_result == 0) {
                    std::lock_guard<std::mutex> lock(insert_latency_mutex);
                    insert_latency_stats.push_back(diff.count() * 1000000); 
                    insert_count++;
                } else {
                    std::cerr << "Insert failed for ID " << event.id << std::endl;
                }
            } else {
                auto qs = std::chrono::high_resolution_clock::now();
                std::vector<TagT> query_result_tags(recall_at);
                std::vector<T *> res;
                index->search_with_tags(event.data.data(), recall_at, Ls, query_result_tags.data(), nullptr, res);
                auto qe = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = qe - qs;
                {
                    std::lock_guard<std::mutex> lock(search_latency_mutex);
                    search_latency_stats.push_back(diff.count() * 1000000); 
                }
                {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    search_results.emplace_back(event.query_idx, query_result_tags);
                }
                search_count++;
            }
        }
    };

    auto start_time = std::chrono::high_resolution_clock::now();
    std::thread producer_thread(producer);
    std::vector<std::thread> workers;
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back(worker, i);
    }

    producer_thread.join();
    for (auto& w : workers) w.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    double insert_qps = insert_count / elapsed_sec;
    double search_qps = search_count / elapsed_sec;
    double insert_qps_per_thread = insert_qps / num_threads;
    double search_qps_per_thread = search_qps / num_threads;

    double mean_insert_latency = insert_latency_stats.empty() ? 0.0 :
        std::accumulate(insert_latency_stats.begin(), insert_latency_stats.end(), 0.0) / insert_latency_stats.size();
    std::sort(insert_latency_stats.begin(), insert_latency_stats.end());
    double p99_insert_latency = insert_latency_stats.empty() ? 0.0 :
        insert_latency_stats[static_cast<size_t>(0.999 * insert_latency_stats.size())];

    double mean_search_latency = search_latency_stats.empty() ? 0.0 :
        std::accumulate(search_latency_stats.begin(), search_latency_stats.end(), 0.0) / search_latency_stats.size();
    std::sort(search_latency_stats.begin(), search_latency_stats.end());
    double p99_search_latency = search_latency_stats.empty() ? 0.0 :
        search_latency_stats[static_cast<size_t>(0.999 * search_latency_stats.size())];

    std::cout << "Total time: " << elapsed_sec << " seconds" << std::endl
              << "Insertion Statistics:" << std::endl
              << "  Ratio: " << write_ratio * 100 << "%" << std::endl
              << "  Overall throughput: " << insert_qps << " points/second" << std::endl
              << "  Per-thread throughput: " << insert_qps_per_thread << " points/second" << std::endl
              << "  Mean latency: " << mean_insert_latency << " microseconds" << std::endl
              << "  P99 latency: " << p99_insert_latency << " microseconds" << std::endl
              << "Search Statistics:" << std::endl
              << "  Ratio: " << (1 - write_ratio) * 100 << "%" << std::endl
              << "  Overall throughput: " << search_qps << " points/second" << std::endl
              << "  Per-thread throughput: " << search_qps_per_thread << " points/second" << std::endl
              << "  Mean latency: " << mean_search_latency << " microseconds" << std::endl
              << "  P99 latency: " << p99_search_latency << " microseconds" << std::endl;

    diskann::aligned_free(data);
    diskann::aligned_free(query);

    const std::string filename = "stats.csv";
    bool file_exists = std::filesystem::exists(filename);
    std::ofstream csv_file(filename, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Error opening CSV file!" << std::endl;
        return false;
    }

    if (!file_exists) {
        csv_file << "Name,Max Connections,ef Build,ef Search,Threads,Write Ratio (%),Write Throughput (points/second),"
                 << "Write Per-thread Throughput (points/second),Write Mean Latency (microseconds),"
                 << "Write P99 Latency (microseconds),Read Ratio (%),Read Throughput (points/second),"
                 << "Read Per-thread Throughput (points/second),Read Mean Latency (microseconds),"
                 << "Read P99 Latency (microseconds)\n";
    }

    csv_file << "pyanns" << "," 
             << R << "," 
             << Lb << "," 
             << Ls << "," 
             << num_threads << ","
             << write_ratio * 100 << ","          // Write Ratio
             << insert_qps << ","                 // Write Overall Throughput
             << insert_qps_per_thread << ","      // Write Per-thread Throughput
             << mean_insert_latency << ","        // Write Mean Latency
             << p99_insert_latency << ","         // Write P99 Latency
             << (1 - write_ratio) * 100 << ","    // Read Ratio
             << search_qps << ","                 // Read Overall Throughput
             << search_qps_per_thread << ","      // Read Per-thread Throughput
             << mean_search_latency << ","        // Read Mean Latency
             << p99_search_latency << "\n";       // Read P99 Latency

    csv_file.close();

    return true;
}

int main(int argc, char **argv)
{
    std::string data_type, query_file, dist_fn, data_path, gt_file, label_file, universal_label, label_type;
    uint32_t num_threads, R, Lb, Ls, build_PQ_bytes, K, batch_size, begin_num;
    float alpha, write_ratio;
    bool use_pq_build, use_opq;
    std::vector<uint32_t> Lvec;

    po::options_description desc{
        program_options_utils::make_program_description("build_memory_index", "Build a memory-based DiskANN index.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                                       program_options_utils::INPUT_DATA_PATH);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("write_ratio", po::value<float>(&write_ratio)->required(), "write ratio");
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);
        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->default_value(10),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        optional_configs.add_options()("batch_size", po::value<uint32_t>(&batch_size)->default_value(100),
                                       "batch size");
        optional_configs.add_options()("begin_num", po::value<uint32_t>(&begin_num)->default_value(1000),
                                       "begin number");
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64),
                                       program_options_utils::MAX_BUILD_DEGREE);
        optional_configs.add_options()("Lbuild,Lb", po::value<uint32_t>(&Lb)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY);
        optional_configs.add_options()("Lsearch,Ls", po::value<uint32_t>(&Ls)->default_value(100),
                                       "graph search complexity");
        optional_configs.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                                       program_options_utils::GRAPH_BUILD_ALPHA);
        optional_configs.add_options()("build_PQ_bytes", po::value<uint32_t>(&build_PQ_bytes)->default_value(0),
                                       program_options_utils::BUIlD_GRAPH_PQ_BYTES);
        optional_configs.add_options()("use_opq", po::bool_switch()->default_value(false),
                                       program_options_utils::USE_OPQ);
        optional_configs.add_options()("label_file", po::value<std::string>(&label_file)->default_value(""),
                                       program_options_utils::LABEL_FILE);
        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                                       program_options_utils::UNIVERSAL_LABEL);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        use_pq_build = (build_PQ_bytes > 0);
        use_opq = vm["use_opq"].as<bool>();
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    try
    {
        if (data_type == "float")
        {
            concurrent_bench<float>(data_path, query_file, begin_num, write_ratio, batch_size, K, Lb, Ls, R, alpha,
                                    num_threads, metric, use_opq, use_pq_build, build_PQ_bytes, gt_file, Lvec);
        }
        else if (data_type == "int8")
        {
            concurrent_bench<int8_t>(data_path, query_file, begin_num, write_ratio, batch_size, K, Lb, Ls, R, alpha,
                                     num_threads, metric, use_opq, use_pq_build, build_PQ_bytes, gt_file, Lvec);
        }
        else if (data_type == "uint8")
        {
            concurrent_bench<uint8_t>(data_path, query_file, begin_num, write_ratio, batch_size, K, Lb, Ls, R, alpha,
                                      num_threads, metric, use_opq, use_pq_build, build_PQ_bytes, gt_file, Lvec);
        }
        else
        {
            std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
            return -1;
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}
