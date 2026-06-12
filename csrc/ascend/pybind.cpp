#include "adapter_meta_kernels.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(kv_cache_adapter_npu_custom_ops, m) {
  m.def("inspect_load_requests", &inspect_load_requests);
  m.def("inspect_save_requests", &inspect_save_requests);
  m.def("pop_reusable_slots", &pop_reusable_slots);
  m.def("commit_load_metadata", &commit_load_metadata);
  m.def("commit_save_metadata", &commit_save_metadata);
  m.def("release_metadata", &release_metadata);
}
