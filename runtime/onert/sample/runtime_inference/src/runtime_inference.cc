/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnfw.h"
#include <csignal>
#include <iostream>
#include <queue>
#include <thread>
#include <vector>
#include <unistd.h>
#include <json.h>
#include <fstream>
#include <unordered_map>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

using namespace std::chrono;

#define PERCENTAGE_THRESHOLD 80
#define PERCENTAGE_THRESHOLD_MEMORY 100
typedef struct MessageQueue
{
  std::queue<std::string> msg_queue;
  pthread_mutex_t msgq_mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t msgq_condition = PTHREAD_COND_INITIALIZER;
} MessageQueue_t;
class JsonWriter;
class ParetoOptimizer;

static MessageQueue_t mq;
static JsonWriter *json;
static volatile int have_request;
static ParetoOptimizer *opt;

class JsonWriter
{
  private:
    Json::StreamWriterBuilder _stream;
    Json::Value _root;
    // Json::Value _vec; _vec(Json::arrayValue),
    std::string _base_filename;
    std::string _dumpfile;
  public:
    JsonWriter(std::string base_filename, std::string dumpfile) :
    _stream(), _root(), _base_filename(base_filename), _dumpfile(dumpfile) {}

    int64_t add_timed_record(std::string name, std::string ph)
    {
      open_file();
      Json::Value rec;
      rec["name"] = name;
      rec["pid"] = 0;
      rec["tid"] = _base_filename;
      rec["ph"] = ph;
      auto ts_val = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
      rec["ts"] = ts_val;
      _root["traceEvents"].append(rec);
      write_to_file();
      return ts_val;
    }

    void add_instance_record(std::string name)
    {
      Json::Value rec;
      open_file();
      rec["name"] = name;
      rec["pid"] = 0;
      rec["tid"] = _base_filename;
      rec["ph"] = "i";
      rec["s"] = "g";              
      rec["ts"] = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
      _root["traceEvents"].append(rec);
      write_to_file();
    }

    void open_file(void)
    {
      std::ifstream _json_input("/tmp/output_trace_"+_dumpfile+".json", std::ifstream::binary);
      if (!_json_input)
      {
        return;
      }
      Json::CharReaderBuilder cfg;
      JSONCPP_STRING errs;
      cfg["collectComments"] = true;
      if (!parseFromStream(cfg, _json_input, &_root, &errs))
      {
        std::cout << errs << std::endl;
      }
      _json_input.close();
    }

    void write_to_file(void)
    {
      std::ofstream _json_file("/tmp/output_trace_"+_dumpfile+".json", std::ofstream::out);
      std::unique_ptr<Json::StreamWriter> writer(_stream.newStreamWriter());      
      writer->write(_root, &_json_file);
      _json_file.close();
    }
};


class ParetoOptimizer
{
  private:
    std::string _json_file;
    std::unordered_map<std::string, std::tuple<float, int32_t>> _solution_map;
    std::unordered_map<std::string, std::string> _backend_allocations;
    float _current_reference_time;
    int32_t _current_reference_memory;
    int32_t _largest_rss_value;
  public:
    ParetoOptimizer(std::string json_file) : _json_file(json_file), _solution_map(), _backend_allocations() {}
    void initialize_maps(void)
    {
      std::ifstream cfg_file(_json_file, std::ifstream::binary);
      Json::CharReaderBuilder cfg;
      Json::Value root;
      Json::Value configs;
      Json::Value solutions;
      JSONCPP_STRING errs;
      cfg["collectComments"] = true;

      if (!parseFromStream(cfg, cfg_file, &root, &errs))
      {
        std::cout << errs << std::endl;
        return;
      }
      configs = root["configs"];
      for (auto it = configs.begin(); it != configs.end(); ++it)
      {
        _backend_allocations[it.key().asString()] = (*it).asString();
      }
      solutions = root["solutions"];
      for (auto it = solutions.begin(); it != solutions.end(); ++it)
      {
        auto solution = *it;
        float exec_time;
        float max_rss;
        _largest_rss_value = 0;
        std::string id;
        for (auto itr = solution.begin(); itr != solution.end(); ++itr)
        {
          if (itr.key() == "id")
          {
            id = (*itr).asString();
          } else if (itr.key() == "exec_time")
          {
            exec_time = (*itr).asFloat();
          } else if (itr.key() == "max_rss")
          {
            max_rss = (*itr).asFloat();
            if (_largest_rss_value < max_rss)
            {
              _largest_rss_value = max_rss;
            }
          }
        }
        _solution_map[id] = std::make_tuple(exec_time, max_rss);
      }
      cfg_file.close();     
    }
    void print_maps(void)
    {
      for (auto x: _solution_map)
      {
        std::cout << x.first << " : (" << std::get<0>(x.second) << "," << std::get<1>(x.second) << ")" << std::endl;
      }
      for (auto x: _backend_allocations)
      {
        std::cout << x.first << " : " << x.second << std::endl;
      }
    }
    float fetch_smallest_exec(void)
    {
      float minval = 999999;
      std::string min_id;
      std::string config;
      for (auto it = _solution_map.begin(); it != _solution_map.end(); ++it)
      {
        auto exec_time = std::get<0>(it->second);
        if (exec_time < minval)
        {
          minval = exec_time;
        }
      }
      return minval;
    }

    std::string fetch_config_with_smallest_exec(void)
    {
      float minval = 999999;
      std::string min_id;
      std::string config;
      for (auto it = _solution_map.begin(); it != _solution_map.end(); ++it)
      {
        auto exec_time = std::get<0>(it->second);
        // std::cout << "Id: " << it->first << " , result = " << std::get<0>(it->second) << std::endl;
        if (exec_time < minval)
        {
          minval = exec_time;
          min_id = it->first;
        }
      }
      _current_reference_time = minval;
      _current_reference_memory = std::get<1>(_solution_map[min_id]);
      config = _backend_allocations[min_id];
      return config;
    }
    std::string fetch_config_with_smallest_memory(void)
    {
      float minval = 9999999;
      std::string min_id;
      std::string config;
      for (auto it = _solution_map.begin(); it != _solution_map.end(); ++it)
      {
        auto max_rss = std::get<1>(it->second);
        // std::cout << "Id: " << it->first << " , result = " << std::get<0>(it->second) << std::endl;
        if (max_rss < minval)
        {
          minval = max_rss;
          min_id = it->first;
        }
      }
      _current_reference_time = std::get<0>(_solution_map[min_id]);
      _current_reference_memory = minval;
      config = _backend_allocations[min_id];
      return config;
    }

    std::string fetch_config_within_exectime(float exec_time_limit)
    {
      float min_difference = 999999;
      float difference;
      std::string min_id;
      std::string config;
      for (auto it = _solution_map.begin(); it != _solution_map.end(); ++it)
      {
        auto exec_time = std::get<0>(it->second);
        if (exec_time < exec_time_limit)
        {
          difference = exec_time_limit - exec_time;
        }
        else
        {
          difference = exec_time - exec_time_limit;
        }
        if (difference < min_difference)
        {
          min_difference = difference;
          min_id = it->first;
        }
      }
      _current_reference_time = std::get<0>(_solution_map[min_id]);
      _current_reference_memory = std::get<1>(_solution_map[min_id]);
      config = _backend_allocations[min_id];
      return config;
    }

    std::string fetch_config_within_memory(int32_t memory_diff)
    {
      float max_val = 0;
      std::string min_id;
      std::string config;
      for (auto it = _solution_map.begin(); it != _solution_map.end(); ++it)
      {
        auto max_rss = std::get<1>(it->second);
        if ((max_rss <= (_current_reference_memory + memory_diff)) && (max_rss > max_val))
        {
          max_val = max_rss;
          min_id = it->first;
        }
      }
      _current_reference_time = std::get<0>(_solution_map[min_id]);
      _current_reference_memory = std::get<1>(_solution_map[min_id]);
      config = _backend_allocations[min_id];
      return config;
    }

    bool exec_time_increased(float exec_time)
    {
      return ((exec_time >  _current_reference_time) && 
              (exec_time - _current_reference_time)*100.0/_current_reference_time > PERCENTAGE_THRESHOLD);
    }

    bool feasible_memory_increase(int32_t memory_diff)
    {
      if (  (memory_diff * 100.0 / _current_reference_memory > PERCENTAGE_THRESHOLD_MEMORY) &&
            (_current_reference_memory != _largest_rss_value) &&
            (_current_reference_memory + memory_diff > _largest_rss_value))
        return true;
      return false;      
    }

    std::string get_current_setting(void)
    {
      return "(" + std::to_string(_current_reference_time) + ", " + std::to_string(_current_reference_memory) + ") ";
    }
};


class RunSession
{
  private:
    std::string _model;
    nnfw_session *_session;
    std::vector<void *> _inputs;
    bool _inputs_initialized;

    uint64_t num_elems(const nnfw_tensorinfo *ti)
    {
      uint64_t n = 1;
      for (uint32_t i = 0; i < ti->rank; ++i)
      {
        n *= ti->dims[i];
      }
      return n;
    }
    template <typename T>
    void random_input_float(float *vec, T nelements)
    {
      T i;
      for (i = 0; i < nelements; i++)
      {
        vec[i] = (((rand() % 100)+1)/1000.f);
      }
    }

    template <typename T1, typename T2>
    void random_input_int(T1 *vec, T2 nelements)
    {
      T2 i;
      for (i = 0; i < nelements; i++)
      {
        vec[i] = rand() % 100 + 1;
      }
    }

  public:
    RunSession(std::string model) : _model(model), _session(nullptr), _inputs(), _inputs_initialized(false) {}

    void load_session(void)
    {
      struct rusage res_usage_begin;
      struct rusage res_usage_end;
      getrusage(RUSAGE_SELF, &res_usage_begin);
      nnfw_create_session(&_session);

      std::string best_config = opt->fetch_config_with_smallest_exec();
      std::string pattern = "OP_BACKEND_MAP=\"";
      std::string backend_setting = best_config.substr(best_config.find(pattern) + pattern.size(), 
                                                      best_config.size() - 1 - (best_config.find(pattern) + pattern.size()) );
      std::cout << "Best Configuration: " << backend_setting << std::endl;
      
      // Loading nnpackage
      nnfw_load_model_from_file(_session, _model.c_str(), backend_setting.c_str());

      // Use acl_neon backend for CONV_2D and acl_cl for otherwise.
      // Note that defalut backend is acl_cl
      // nnfw_set_available_backends(session, "acl_cl;cpu");
      // nnfw_set_available_backends(session, "cpu");
      // nnfw_set_op_backend(session, "CONV_2D", "cpu");  
      // nnfw_set_op_backend(session, "BATCH_TO_SPACE_ND", "cpu");  
      // Compile model
      nnfw_prepare(_session);
      getrusage(RUSAGE_SELF, &res_usage_end);
      json->add_instance_record("Initial setting: (" + get_pareto_setting() + 
                            "), RSS increase: (" + std::to_string(res_usage_end.ru_maxrss) + ", " + std::to_string(res_usage_begin.ru_maxrss)+")");
    }

    bool latency_increased(float exec_time)
    {
      return opt->exec_time_increased(exec_time);
    }

    bool memory_improved(int32_t memory_diff)
    {
      return opt->feasible_memory_increase(memory_diff);
    }

    void reconfigure_within_exec_time(float exec_time)
    {
      std::string setting_old = get_pareto_setting();
      std::string best_config = opt->fetch_config_within_exectime(exec_time);
      // std::string best_config = opt->fetch_config_with_smallest_memory();
      std::string pattern = "OP_BACKEND_MAP=\"";
      std::string backend_setting = best_config.substr(best_config.find(pattern) + pattern.size(), 
                                                      best_config.size() - 1 - (best_config.find(pattern) + pattern.size()) );
      std::cout << "new backend setting for model " << _model << ": " << backend_setting << std::endl;
      json->add_timed_record("session reconfig", "B");
      nnfw_close_session(_session);
      struct rusage res_usage_begin;
      struct rusage res_usage_end;
      getrusage(RUSAGE_SELF, &res_usage_begin);
      nnfw_create_session(&_session);
      nnfw_load_model_from_file(_session, _model.c_str(), backend_setting.c_str());
      nnfw_prepare(_session);
      getrusage(RUSAGE_SELF, &res_usage_end);
      json->add_instance_record(setting_old + " -> Alert increased time -> " + get_pareto_setting() +
                                          "RSS increase: (" + std::to_string(res_usage_end.ru_maxrss) + ", " + std::to_string(res_usage_begin.ru_maxrss)+")");
      prepare_output();

      json->add_timed_record("session reconfig", "E");
    }


    void reconfigure_within_memory(int32_t memory_val)
    {
      std::string setting_old = get_pareto_setting();
      std::string best_config = opt->fetch_config_within_memory(memory_val);
      // std::string best_config = opt->fetch_config_with_smallest_memory();
      std::string pattern = "OP_BACKEND_MAP=\"";
      std::string backend_setting = best_config.substr(best_config.find(pattern) + pattern.size(), 
                                                      best_config.size() - 1 - (best_config.find(pattern) + pattern.size()) );
      std::cout << "new backend setting for model " << _model << ": " << backend_setting << std::endl;
      json->add_timed_record("session reconfig", "B");
      nnfw_close_session(_session);
      struct rusage res_usage_begin;
      struct rusage res_usage_end;
      getrusage(RUSAGE_SELF, &res_usage_begin);
      nnfw_create_session(&_session);
      nnfw_load_model_from_file(_session, _model.c_str(), backend_setting.c_str());
      nnfw_prepare(_session);
      getrusage(RUSAGE_SELF, &res_usage_end);
      prepare_output();
      json->add_instance_record(setting_old + " --> " + get_pareto_setting() +
                                "RSS increase: (" + std::to_string(res_usage_end.ru_maxrss) + ", " + std::to_string(res_usage_begin.ru_maxrss) + ")");
      json->add_timed_record("session reconfig", "E");
    }

    int64_t run_inference(void)
    {
      int64_t st_time;
      int64_t end_time;
      st_time = json->add_timed_record("session run", "B");
      nnfw_run(_session);
      end_time = json->add_timed_record("session run", "E");
      return (end_time - st_time);
    }

    void close()
    {
      nnfw_close_session(_session);
    }

    void initialize_inputs(void)
    {
      uint32_t n_inputs;
      nnfw_tensorinfo ti;
      nnfw_input_size(_session, &n_inputs);
      for (auto i = 0; i < n_inputs; i++)
      {        
        nnfw_input_tensorinfo(_session, i, &ti); // get first input's info
        uint32_t input_elements = num_elems(&ti);
        switch (ti.dtype)
        {
          case NNFW_TYPE_TENSOR_FLOAT32:
          {
            float *input;         
            if (_inputs_initialized == false)
            {
              input = new float[input_elements];              
              _inputs.emplace_back(static_cast<void *>(input));
            }
            else
            {
              input = static_cast<float *>(_inputs[i]);
            }
            random_input_float(input, input_elements);
            nnfw_set_input(_session, i, ti.dtype, input, sizeof(float) * input_elements);
            break;
          }
          case NNFW_TYPE_TENSOR_INT32:
          {            
            int32_t *input;
            random_input_int(input, input_elements);
            if (_inputs_initialized == false)
            {
              input = new int32_t[input_elements];              
              _inputs.emplace_back(static_cast<void *>(input));
            }
            else
            {
              input = static_cast<int32_t *>(_inputs[i]);
            }
            random_input_int<int32_t, uint32_t>(input, input_elements);
            nnfw_set_input(_session, i, ti.dtype, input, sizeof(int32_t) * input_elements);
            break;
          }
          case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
          case NNFW_TYPE_TENSOR_BOOL:
          case NNFW_TYPE_TENSOR_UINT8:
          {
            uint8_t *input;
            if (_inputs_initialized == false)
            {
              input = new uint8_t[input_elements];              
              _inputs.emplace_back(static_cast<void *>(input));
            }
            else
            {
              input = static_cast<uint8_t *>(_inputs[i]);
            }
            random_input_int<uint8_t, uint32_t>(input, input_elements);
            nnfw_set_input(_session, i, ti.dtype, input, sizeof(uint8_t) * input_elements);
            break;
          }
          case NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
          { 
            int8_t *input;
            if (_inputs_initialized == false)
            {
              int8_t *input = new int8_t[input_elements];              
              _inputs.emplace_back(static_cast<void *>(input));
            }
            else
            {
              input = static_cast<int8_t *>(_inputs[i]);
            }
            random_input_int<int8_t, uint32_t>(input, input_elements);
            nnfw_set_input(_session, i, ti.dtype, input, sizeof(int8_t) * input_elements);
            break;
          }
          case NNFW_TYPE_TENSOR_INT64:
          {
            int64_t *input;
            if (_inputs_initialized == false)
            {
              int64_t *input = new int64_t[input_elements];              
              _inputs.emplace_back(static_cast<void *>(input));
            }
            else
            {
              input = static_cast<int64_t *>(_inputs[i]);
            }
            random_input_int<int64_t, uint32_t>(input, input_elements);
            nnfw_set_input(_session, i, ti.dtype, input, sizeof(int64_t) * input_elements);
            break;
          }
          default:
            std::cout << "Uknown input data type " << ti.dtype << std::endl;
            break;    
        }
      }
      if (_inputs_initialized == false)
      {
        _inputs_initialized = true;
      }
    }
    

    void prepare_output(void)
    {
      uint32_t n_outputs;
      nnfw_tensorinfo ti;
      nnfw_output_size(_session, &n_outputs);
      for (auto i = 0; i < n_outputs; i++)
      {
        nnfw_output_tensorinfo(_session, i, &ti); // get first output's info
        uint32_t output_elements = num_elems(&ti);    
        switch (ti.dtype)
        {
          case NNFW_TYPE_TENSOR_FLOAT32:
          {
            float *output = new float[output_elements];
            nnfw_set_output(_session, i, ti.dtype, output, sizeof(float) * output_elements);            
            break;
          }
          case NNFW_TYPE_TENSOR_INT32:
          {
            int32_t *output = new int32_t[output_elements];
            nnfw_set_output(_session, i, ti.dtype, output, sizeof(int32_t) * output_elements);            
            break;
          }
          case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
          case NNFW_TYPE_TENSOR_BOOL:
          case NNFW_TYPE_TENSOR_UINT8:
          {
            uint8_t *output = new uint8_t[output_elements];
            nnfw_set_output(_session, i, ti.dtype, output, sizeof(uint8_t) * output_elements);            
            break;
          }
          case NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
          { 
            int8_t *output = new int8_t[output_elements];
            nnfw_set_output(_session, i, ti.dtype, output, sizeof(int8_t) * output_elements);            
            break;
          }
          case NNFW_TYPE_TENSOR_INT64:
          {
            int64_t *output = new int64_t[output_elements];
            nnfw_set_output(_session, i, ti.dtype, output, sizeof(int64_t) * output_elements);            
            break;
          }
          default:
            std::cout << "Uknown output data type " << ti.dtype << std::endl;
            break;    
        }    
      }
    }

    std::string get_pareto_setting(void)
    {
      return opt->get_current_setting();
    }
};


void signalHandler( int signum )
{
  std::cout << "Interrupt signal (" << signum << ") received." << std::endl;

  // cleanup and close up stuff here
  // terminate program
  json->add_instance_record("Thread Exit");
  json->write_to_file();
  exit(signum);
}

void initialize_globals(std::string model, std::string config_file, std::string dumpfile)
{  
  std::string base_filename = model.substr(model.find_last_of("/\\") + 1);
  json = new JsonWriter(base_filename, dumpfile);
  opt = new ParetoOptimizer(config_file);
  opt->initialize_maps();
  // opt->print_maps();  
}


unsigned long get_available_memory(void)
{
  std::string token;
  std::ifstream file("/proc/meminfo");
  while(file >> token) {
      if(token == "MemAvailable:") {
          unsigned long mem;
          if(file >> mem) {
              return mem;
          } else {
              return 0;       
          }
      }
      // ignore rest of the line
      file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return 0; // nothing found
}

void runtime_thread(std::string model)
{  
  // nnfw_session *session = nullptr;
  struct rusage res_usage_begin;
  struct rusage res_usage_end;
  json->add_timed_record("Runtime", "B");
  auto available_memory = get_available_memory();
  int32_t reference_memory = available_memory;
  std::cout << "Available memory before session creation: " << available_memory << std::endl;
  // getrusage(RUSAGE_SELF, &res_usage_begin);
  json->add_timed_record("session prepare", "B");
  
  // Prepare session.  
  RunSession my_session(model);
  my_session.load_session();

  // Prepare output
  my_session.prepare_output();
  json->add_timed_record("session prepare", "E");
  // getrusage(RUSAGE_SELF, &res_usage_end);
  std::cout << "model loaded" << std::endl;
  // json->add_instance_record("Initial setting: (" + my_session.get_pareto_setting() + 
  //                           "), RSS increase: (" + std::to_string(res_usage_end.ru_maxrss) + ", " + std::to_string(res_usage_begin.ru_maxrss)+")");
  pthread_mutex_lock(&mq.msgq_mutex);
  mq.msg_queue.push("infer");
  pthread_mutex_unlock(&mq.msgq_mutex);
  pthread_cond_signal(&mq.msgq_condition);
  int inference_cnt = 0;  
  float exec_time;
  while(1)
  {
      pthread_mutex_lock(&mq.msgq_mutex);
      pthread_cond_wait(&mq.msgq_condition, &mq.msgq_mutex);
      pthread_mutex_unlock(&mq.msgq_mutex);
      std::cout << "thread received signal" << std::endl;

      if(!mq.msg_queue.empty())
      {
          std::string s = mq.msg_queue.front();
          mq.msg_queue.pop();
          std::cout << "received msg " << s << std::endl;          
          if (s == "infer")
          {
              // Initialize inputs.
              my_session.initialize_inputs();
              // Do inference
              exec_time = my_session.run_inference();
              std::cout << "Inference iteration: " << inference_cnt++ << " done" << std::endl;
              exec_time /= 1000.0;
              
              if (my_session.latency_increased(exec_time))
              {
                // std::cout << "ALERT: execution time increased from " << reference_exec_time << " to " << exec_time << " ms" << std::endl;
                // std::string setting_old = my_session.get_pareto_setting();
                // getrusage(RUSAGE_SELF, &res_usage_begin);
                my_session.reconfigure_within_exec_time(exec_time);
                // std::string setting_new = my_session.get_pareto_setting();
                // getrusage(RUSAGE_SELF, &res_usage_end);
                // json->add_instance_record(setting_old + " -> Alert increased time -> " + setting_new +
                //                           "RSS increase: (" + std::to_string(res_usage_end.ru_maxrss) + ", " + std::to_string(res_usage_begin.ru_maxrss)+")");
              }
              available_memory = get_available_memory();
              if ((available_memory > reference_memory) && my_session.memory_improved(available_memory - reference_memory))
              {
                // std::string setting_old = my_session.get_pareto_setting();
                // getrusage(RUSAGE_SELF, &res_usage_begin);                
                my_session.reconfigure_within_memory(available_memory - reference_memory);
                // std::string setting_new = my_session.get_pareto_setting();
                // getrusage(RUSAGE_SELF, &res_usage_end);
                // std::string rec = " -> Alert better memory: (" + std::to_string(available_memory) + ", " + std::to_string(reference_memory) + ") ->";
                // json->add_instance_record(setting_old + rec + setting_new +
                //                           "RSS increase: (" + std::to_string(res_usage_end.ru_maxrss) + ", " + std::to_string(res_usage_begin.ru_maxrss) + ")");
              }
              reference_memory = available_memory;
              while (!have_request);
              // pthread_mutex_lock(&mq.msgq_mutex);              
              mq.msg_queue.push("inferDone");          
              pthread_cond_signal(&mq.msgq_condition);
              // pthread_mutex_unlock(&mq.msgq_mutex);
              have_request = 0;
          }
          else if (s == "exit")
          {
              // pthread_mutex_unlock(&mq.msgq_mutex);
              break;
          }
          else
          {
              std::cout << "unknown message " << s << std::endl;
          }
      }
  }

  my_session.close();


  json->add_timed_record("Runtime", "E");
  json->write_to_file();
  std::cout << "nnpackage " << model << " runs successfully." << std::endl;
}

int main(const int argc, char **argv)
{
  srand(time(NULL));
  signal(SIGINT, signalHandler);
  signal(SIGKILL, signalHandler);
  signal(SIGTERM, signalHandler);
  initialize_globals(argv[1], argv[2], argv[3]);
 
  std::thread runtime(runtime_thread, argv[1]);
  pthread_cond_wait(&mq.msgq_condition, &mq.msgq_mutex);
  mq.msg_queue.pop();
  pthread_mutex_unlock(&mq.msgq_mutex);
  auto n_iterations = std::stoi(argv[4]);
  for (auto i = 0; i < n_iterations; i++)
  {    
    while (have_request) ;
    // pthread_mutex_lock(&mq.msgq_mutex);
    mq.msg_queue.push("infer");
    pthread_cond_signal(&mq.msgq_condition);
    // pthread_mutex_unlock(&mq.msgq_mutex);
    have_request = 1;
    pthread_mutex_lock(&mq.msgq_mutex);
    pthread_cond_wait(&mq.msgq_condition, &mq.msgq_mutex);
    pthread_mutex_unlock(&mq.msgq_mutex);
    mq.msg_queue.pop();    
  }

  std::cout << "main calling runtime to exit.." << std::endl;
  std::string msg = "exit";
  pthread_mutex_lock(&mq.msgq_mutex);
  mq.msg_queue.push(msg);
  pthread_cond_signal(&mq.msgq_condition);
  pthread_mutex_unlock(&mq.msgq_mutex);
  std::cout << "main sent signal" << std::endl;
  runtime.join();
  std::cout << "main exiting" << std::endl;
  return 0;
}
