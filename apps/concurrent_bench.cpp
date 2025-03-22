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
                  << " #threads: " << num_threads << " #ratio: " << write_ratio << ":" << 1 - write_ratio << std::endl;

    size_t data_num, data_dim, aligned_dim;
    diskann::get_bin_metadata(data_path, data_num, data_dim);

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

    T *query = nullptr;
    size_t query_num, query_dim, query_aligned_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    T *data = nullptr;
    diskann::load_aligned_bin<T>(data_path, data, data_num, data_dim, aligned_dim);

    // build with begin size
    std::vector<uint32_t> tags(begin_num);
    std::iota(tags.begin(), tags.end(), 1 + static_cast<uint32_t>(0));
    index->build(data, begin_num, tags);

    // concurrent read and write
    size_t insert_total = data_num - begin_num;
    size_t search_total = insert_total * ((1 - write_ratio) / write_ratio);
    size_t search_batch_size = batch_size * ((1 - write_ratio) / write_ratio);
    size_t start_insert_offset = 0, end_insert_offset = 0, start_search_offset = 0, end_search_offset = 0,
           query_idx = 0;

    std::exception_ptr last_exception = nullptr;
    std::mutex last_except_mutex, result_mutex, insert_latency_mutex, search_latency_mutex;
    std::vector<double> insert_latency_stats, search_latency_stats;
    std::vector<SearchResult<uint32_t>> search_results;
    diskann::ThreadPool pool(num_threads);

    auto succeed_insert_count = std::make_shared<std::atomic<size_t>>(0);
    auto failed_insert_count = std::make_shared<std::atomic<size_t>>(0);
    auto succeed_search_count = std::make_shared<std::atomic<size_t>>(0);
    auto failed_search_count = std::make_shared<std::atomic<size_t>>(0);

    auto st = std::chrono::high_resolution_clock::now();
    while (end_insert_offset < insert_total || end_search_offset < search_total)
    {
        end_insert_offset = std::min(start_insert_offset + batch_size, insert_total);
        for (size_t idx = start_insert_offset; idx < end_insert_offset; ++idx)
        {
            pool.enqueue_task([&, idx] {
                try
                {
                    auto qs = std::chrono::high_resolution_clock::now();

                    int insert_result = index->insert_point(&data[(idx + begin_num) * aligned_dim],
                                                            1 + static_cast<TagT>(idx + begin_num));
                    if (insert_result != 0)
                    {
                        failed_insert_count->fetch_add(1, std::memory_order_seq_cst);
                        std::cerr << "Insert failed " << idx << std::flush;
                    }
                    else
                        succeed_insert_count->fetch_add(1, std::memory_order_seq_cst);

                    auto qe = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> diff = qe - qs;
                    {
                        std::unique_lock<std::mutex> lock(insert_latency_mutex);
                        insert_latency_stats.push_back((float)(diff.count() * 1000000));
                    }
                }
                catch (...)
                {
                    std::unique_lock<std::mutex> lock(last_except_mutex);
                    last_exception = std::current_exception();
                }
            });
        }
        start_insert_offset = end_insert_offset;

        end_search_offset = std::min(start_search_offset + search_batch_size, search_total);
        for (size_t idx = start_search_offset; idx < end_search_offset; ++idx)
        {
            if (++query_idx >= query_num)
                query_idx %= query_num;
            pool.enqueue_task([&, query_idx] {
                auto qs = std::chrono::high_resolution_clock::now();
                try
                {
                    auto qs = std::chrono::high_resolution_clock::now();

                    std::vector<TagT> query_result_tags(recall_at);
                    std::vector<T *> res;
                    index->search_with_tags(query + query_idx * query_aligned_dim, recall_at, Ls, query_result_tags.data(),
                                            nullptr, res);

                    auto qe = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> diff = qe - qs;
                    {
                        std::unique_lock<std::mutex> lock(search_latency_mutex);
                        search_latency_stats.push_back((float)(diff.count() * 1000000));
                    }
                    {
                        std::unique_lock<std::mutex> lock(result_mutex);
                        search_results.emplace_back(end_insert_offset, query_idx, query_result_tags);
                    }
                }
                catch (...)
                {
                    std::unique_lock<std::mutex> lock(last_except_mutex);
                    last_exception = std::current_exception();
                }
            });
        }
        start_search_offset = end_search_offset;
    }

    pool.wait_for_tasks();

    auto et = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(et - st).count();

    double insert_qps = insert_total / elapsed_sec;
    double search_qps = search_total / elapsed_sec;

    double insert_qps_per_thread = insert_total / elapsed_sec / num_threads;
    double search_qps_per_thread = search_total / elapsed_sec / num_threads;

    if (last_exception)
    {
        std::rethrow_exception(last_exception);
    }

    std::sort(insert_latency_stats.begin(), insert_latency_stats.end());
    double mean_insert_latency = std::accumulate(insert_latency_stats.begin(), insert_latency_stats.end(), 0.0) /
                                 static_cast<float>(insert_total);
    double p99_insert_latency =
        insert_latency_stats.empty() ? 0.0 : (float)insert_latency_stats[(uint64_t)(0.999 * insert_total)];

    std::sort(search_latency_stats.begin(), search_latency_stats.end());
    double mean_search_latency = std::accumulate(search_latency_stats.begin(), search_latency_stats.end(), 0.0) /
                                 static_cast<float>(search_total);
    double p99_search_latency =
        search_latency_stats.empty() ? 0.0 : (float)search_latency_stats[(uint64_t)(0.999 * search_total)];

    std::cout << "Total time: " << elapsed_sec << " seconds" << std::endl
              << "Insertion Statistics:" << std::endl
              << "  Ratio: " << write_ratio * 100 << "%" << std::endl
              << "  Overall throughput: " << insert_qps << " points/second" << std::endl
              << "  Per-thread throughput: " << insert_qps_per_thread << " points/second" << std::endl
              << "  Mean latency: " << (insert_latency_stats.empty() ? 0.0 : mean_insert_latency) << " microseconds"
              << std::endl
              << "  P99 latency: " << (insert_latency_stats.empty() ? 0.0 : p99_insert_latency) << " microseconds"
              << std::endl
              << std::endl
              << "Search Statistics:" << std::endl
              << "  Total time: " << (1 - write_ratio) * 100 << "%" << std::endl
              << "  Overall throughput: " << search_qps << " points/second" << std::endl
              << "  Per-thread throughput: " << search_qps_per_thread << " points/second" << std::endl
              << "  Mean latency: " << (search_latency_stats.empty() ? 0.0 : mean_search_latency) << " microseconds"
              << std::endl
              << "  P99 latency: " << (search_latency_stats.empty() ? 0.0 : p99_search_latency) << " microseconds"
              << std::endl
              << std::endl;

    diskann::aligned_free(data);
    diskann::aligned_free(query);

    const std::string filename = "stats.csv";
    bool file_exists = std::filesystem::exists(filename); // 检查文件是否存在
    std::ofstream csv_file(filename, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Error opening CSV file!" << std::endl;
        return false;
    }

    if (!file_exists) {
        csv_file << "Name,Threads,Write Ratio (%),Write Throughput (points/second),"
                 << "Write Per-thread Throughput (points/second),Write Mean Latency (microseconds),"
                 << "Write P99 Latency (microseconds),Read Ratio (%),Read Throughput (points/second),"
                 << "Read Per-thread Throughput (points/second),Read Mean Latency (microseconds),"
                 << "Read P99 Latency (microseconds)\n";
    }

    csv_file << "diskann" << "," 
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

    // overall recall
//     std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
//     std::vector<float> latency_stats(query_num, 0);
//     uint32_t *gt_ids = nullptr;
//     float *gt_dists = nullptr;
//     size_t gt_num, gt_dim;
//     uint32_t table_width = 0;
//     const uint32_t first_recall = false ? 1 : recall_at;
//     std::vector<TagT> query_result_tags(recall_at * query_num);
//     uint32_t recalls_to_print = recall_at + 1 - first_recall;
//     table_width += recalls_to_print * 12;
//     double best_recall = 0.0;

//     if (truthset_file != std::string("null") && file_exists(truthset_file))
//     {
//         diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
//         if (gt_num != query_num)
//         {
//             std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
//         }
//     }

//     std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS" << std::setw(20) << "Mean Latency (mus)"
//                 << std::setw(15) << "99.9 Latency";
//     table_width += 4 + 12 + 20 + 15;
//     for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
//     {
//         std::cout << std::setw(12) << ("Recall@" + std::to_string(curr_recall));
//     }
//     recalls_to_print = recall_at + 1 - first_recall;
//     table_width += recalls_to_print * 12;
//     std::cout << std::endl;
//     std::cout << std::string(table_width, '=') << std::endl;

//     for (uint32_t L_id = 0; L_id < Lvec.size(); L_id++)
//     {
//         uint32_t L = Lvec[L_id];
//         if (L < recall_at)
//         {
//             diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
//             continue;
//         }

//         query_result_ids[L_id].resize(recall_at * query_num);
//         std::vector<T *> res = std::vector<T *>();
//         auto s = std::chrono::high_resolution_clock::now();
//         omp_set_num_threads(num_threads);
// #pragma omp parallel for schedule(dynamic, 1)
//         for (int64_t i = 0; i < (int64_t)query_num; i++)
//         {
//             auto qs = std::chrono::high_resolution_clock::now();
//             index->search_with_tags(query + i * query_aligned_dim, recall_at, L,
//                                     query_result_tags.data() + i * recall_at, nullptr, res);

//             auto qe = std::chrono::high_resolution_clock::now();
//             std::chrono::duration<double> diff = qe - qs;
//             latency_stats[i] = (float)(diff.count() * 1000000);
//         }
//         std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

//         double displayed_qps = query_num / diff.count();
//         displayed_qps /= num_threads;

//         std::vector<double> recalls;
//         recalls.reserve(recalls_to_print);
//         for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
//         {
//             recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
//                                                         query_result_ids[L_id].data(), recall_at, curr_recall));
//         }

//         std::sort(latency_stats.begin(), latency_stats.end());
//         double mean_latency =
//             std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

//         std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(20) << (float)mean_latency
//                     << std::setw(15) << (float)latency_stats[(uint64_t)(0.999 * query_num)];

//         for (double recall : recalls)
//         {
//             std::cout << std::setw(12) << recall;
//             best_recall = std::max(recall, best_recall);
//         }
//         std::cout << std::endl;
//     }
    
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
        optional_configs.add_options()("begin_num", po::value<uint32_t>(&begin_num)->default_value(5000),
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
