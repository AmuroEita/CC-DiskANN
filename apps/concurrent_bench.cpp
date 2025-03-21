// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <cstring>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"
#include "timer.h"
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

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
bool concurrent_bench(const std::string data_path, const std::string &query_file, const size_t begin_num, 
                      const uint32_t L, const uint32_t R, const float alpha, const uint32_t num_threads, 
                      const diskann::Metric metric, const bool use_opq, const bool use_pq_build, 
                      const uint32_t build_PQ_bytes, const size_t batch_size, const uint32_t recall_at)
{
    size_t data_num, data_dim, aligned_dim;
    diskann::get_bin_metadata(data_path, data_num, data_dim);

    size_t query_num, query_dim, query_aligned_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
                                .with_filter_list_size(Lf)
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
                        .with_data_type(diskann_type_to_name<T>)
                        .with_label_type(diskann_type_to_name<labelT>)
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
    aligned_dim = ROUND_UP(data_dim, 8);
    diskann::alloc_aligned((void **)&data, data_num * aligned_dim * sizeof(float),
                            8 * sizeof(float));

    // build with begin size
    load_aligned_bin_part(data_path, data, 0, begin_size);
    std::vector<uint32_t> tags(begin_size);
    std::iota(tags.begin(), tags.end(), 1 + static_cast<uint32_t>(0));
    index->build(data, begin_size, tags);

    // concurrent read and write
    size_t insert_total = data_num - begin_size;
    size_t search_total = insert_total * ((1 - write_ratio) / write_ratio);
    size_t qtSize = qt.size(0);
    size_t search_batch_size = batch_size * ((1 - write_ratio) / write_ratio);

    std::atomic<size_t> next_insert_idx = 0, next_search_idx = 0;
    std::atomic<size_t> completed_inserts = 0, completed_searches = 0;

    std::exception_ptr last_exception = nullptr;
    std::mutex last_except_mutex, result_mutex, insert_latency_mutex, search_latency_mutex;

    std::vector<double> insert_latency_stats, search_latency_stats;

    INTELLI::ThreadPool pool(num_threads);
    load_aligned_bin_part(data_path, data, begin_size, data_num - begin_size);

    diskann::Timer timer;
    while (next_insert_idx < insert_total || next_search_idx < search_total)
    {
        size_t start_insert_offset = next_insert_idx + batch_size;
        size_t end_insert_offset = std::min(start_insert_offset + batch_size, insert_total);

        for (size_t idx = start_insert_offset; idx < end_insert_offset; ++idx)
        {
            pool.enqueue_task([&, idx] {
                try
                {
                    auto qs = std::chrono::high_resolution_clock::now();
                  
                    int insert_result = -1;
                    insert_result = index->insert_point(&data[(idx - begin_size) * aligned_dim], 1 + static_cast<TagT>(idx));
                    if (insert_result != 0) std::cerr << "Insert failed " << idx << std::endl;

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
        
        size_t start_search_offset = next_search_idx + search_batch_size;
        size_t end_search_offset = std::min(start_search + search_batch_size, search_total);

        size_t query_idx = 0;
        for (size_t idx = start_search_offset; idx < end_search_offset; ++idx, ++query_idx)
        {
            if (query_idx >= qeury_num) 
                qeury_idx %= query_num;
            pool.enqueue_task([&, qeury_idx] {
                auto qs = std::chrono::high_resolution_clock::now();
                try
                {
                    auto qs = std::chrono::high_resolution_clock::now();
                  
                    std::vector<TagT> query_result_tags(recall_at);
                    std::vector<T *> res
                    index->search_with_tags(query + idx * query_aligned_dim, recall_at, L, 
                                            query_result_tags.data(), nullptr, res);

                    auto qe = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> diff = qe - qs;
                    {
                        std::unique_lock<std::mutex> lock(search_latency_mutex);
                        searchLatencies.push_back((float)(diff.count() * 1000000));
                    }
                    {
                        std::unique_lock<std::mutex> lock(resultMutex);
                        searchRes.emplace_back(end_insert_offset, query_idx, query_result_tags);
                    }
                }
                catch (...)
                {
                    std::unique_lock<std::mutex> lock(last_except_mutex);
                    last_exception = std::current_exception();
                }
            });
        }
    }

    pool.wait_for_tasks();

    auto et = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(et - st).count();

    // ROUND UP
    double insert_qps = insert_total / elapsed_sec;
    double search_qps = search_total / elapsed_sec;

    double insert_qps_per_thread = insert_total / elapsed_sec / num_threads;
    double search_qps_per_thread = search_total / elapsed_sec / num_threads;

    if (last_exception)
    {
        std::rethrow_exception(last_exception);
    }
  
    std::sort(insert_latency_stats.begin(), insert_latency_stats.end());
    double mean_insert_latency =
        std::accumulate(insert_latency_stats.begin(), insert_latency_stats.end(), 0.0) / static_cast<float>(insert_total);
    double p99_insert_latency = (float)insert_latency_stats[(uint64_t)(0.999 * insert_total)];

    std::sort(search_latency_stats.begin(), search_latency_stats.end());
    double mean_search_latency =
        std::accumulate(search_latency_stats.begin(), search_latency_stats.end(), 0.0) / static_cast<float>(insert_total);
    double p99_search_latency = (float)search_latency_stats[(uint64_t)(0.999 * search_total)];

    return true;
}

template <typename T>
inline void load_aligned_bin_part(const std::string &bin_file, T *data, size_t offset_points, size_t points_to_read)
{
    diskann::Timer timer;
    std::ifstream reader;
    reader.exceptions(std::ios::failbit | std::ios::badbit);
    reader.open(bin_file, std::ios::binary | std::ios::ate);
    size_t actual_file_size = reader.tellg();
    reader.seekg(0, std::ios::beg);

    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    size_t npts = (uint32_t)npts_i32;
    size_t dim = (uint32_t)dim_i32;

    size_t expected_actual_file_size = npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size)
    {
        std::stringstream stream;
        stream << "Error. File size mismatch. Actual size is " << actual_file_size << " while expected size is  "
               << expected_actual_file_size << " npts = " << npts << " dim = " << dim << " size of <T>= " << sizeof(T)
               << std::endl;
        std::cout << stream.str();
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (offset_points + points_to_read > npts)
    {
        std::stringstream stream;
        stream << "Error. Not enough points in file. Requested " << offset_points << "  offset and " << points_to_read
               << " points, but have only " << npts << " points" << std::endl;
        std::cout << stream.str();
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    reader.seekg(2 * sizeof(uint32_t) + offset_points * dim * sizeof(T));

    const size_t rounded_dim = ROUND_UP(dim, 8);

    for (size_t i = 0; i < points_to_read; i++)
    {
        reader.read((char *)(data + i * rounded_dim), dim * sizeof(T));
        memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
    reader.close();

    const double elapsedSeconds = timer.elapsed() / 1000000.0;
    std::cout << "Read " << points_to_read << " points using non-cached reads in " << elapsedSeconds << std::endl;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, data_path, index_path_prefix, label_file, universal_label, label_type;
    uint32_t num_threads, R, L, Lf, build_PQ_bytes;
    float alpha;
    bool use_pq_build, use_opq;

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
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                                       program_options_utils::INPUT_DATA_PATH);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64),
                                       program_options_utils::MAX_BUILD_DEGREE);
        optional_configs.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY);
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

        optional_configs.add_options()("FilteredLbuild", po::value<uint32_t>(&Lf)->default_value(0),
                                       program_options_utils::FILTERED_LBUILD);
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
        diskann::cout << "Starting index build with R: " << R << "  Lbuild: " << L << "  alpha: " << alpha
                      << "  #threads: " << num_threads << std::endl;

        size_t data_num, data_dim, aligned_dim;
        size_t begin_size = 1000;
        diskann::get_bin_metadata(data_path, data_num, data_dim);

        auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
                                      .with_filter_list_size(Lf)
                                      .with_alpha(alpha)
                                      .with_saturate_graph(false)
                                      .with_num_threads(num_threads)
                                      .build();

        auto filter_params = diskann::IndexFilterParamsBuilder()
                                 .with_universal_label(universal_label)
                                 .with_label_file(label_file)
                                 .with_save_path_prefix(index_path_prefix)
                                 .build();
        auto config = diskann::IndexConfigBuilder()
                          .with_metric(metric)
                          .with_dimension(data_dim)
                          .with_max_points(data_num)
                          .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                          .with_data_type(data_type)
                          .with_label_type(label_type)
                          .is_dynamic_index(false)
                          .with_index_write_params(index_build_params)
                          .is_enable_tags(false)
                          .is_use_opq(use_opq)
                          .is_pq_dist_build(use_pq_build)
                          .with_num_pq_chunks(build_PQ_bytes)
                          .build();

        auto index_factory = diskann::IndexFactory(config);
        auto index = index_factory.create_instance();

        if (data_type == "float") {
            float *data = nullptr;
            aligned_dim = ROUND_UP(data_dim, 8);
            diskann::alloc_aligned((void **)&data, data_num * aligned_dim * sizeof(float),
                            8 * sizeof(float));
            load_aligned_bin_part(data_path, data, 0, begin_size);

            std::vector<uint32_t> tags(begin_size);
            std::iota(tags.begin(), tags.end(), 1 + static_cast<uint32_t>(0));
            index->build(data, begin_size, tags);
            
            load_aligned_bin_part(data_path, data, begin_size, data_num - begin_size);

            size_t num_failed = 0;
            diskann::Timer insert_timer;

#pragma omp parallel for num_threads((int32_t)num_threads) schedule(dynamic) reduction(+ : num_failed)
            for (int64_t j = begin_size; j < (int64_t)data_num; j++)
            {
                int insert_result = -1;
                insert_result = index->insert_point(&data[(j - begin_size) * aligned_dim], 1 + static_cast<uint32_t>(j));
            }

            const double elapsedSeconds = insert_timer.elapsed() / 1000000.0;
            std::cout << "Cores " << num_threads << " Insertion time " << elapsedSeconds << " seconds (" << (data_num - begin_size) / elapsedSeconds
                    << " points/second overall, " << (data_num - begin_size) / elapsedSeconds / num_threads << " per thread)\n ";
        }
        
        // index->build(data_path, data_num, filter_params);
        // index->save(index_path_prefix.c_str());
        index.reset();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}

