#include "adapter_meta_kernels.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(kv_cache_adapter_npu_custom_ops, m) {
  m.def("inspect_load_requests", &inspect_load_requests);
  m.def("inspect_save_requests", &inspect_save_requests);
  m.def("pop_reusable_slots", &pop_reusable_slots);
  m.def("_debug_mark_blocked_slots", &debug_mark_blocked_slots);
  m.def("_debug_count_threshold_slots", &debug_count_threshold_slots);
  m.def("_debug_plan_threshold_slots", &debug_plan_threshold_slots);
  m.def("_debug_collect_threshold_slots", &debug_collect_threshold_slots);
  m.def("_debug_age_usage", &debug_age_usage);
  m.def("_debug_finalize_selected_slots", &debug_finalize_selected_slots);
  m.def("commit_load_metadata", &commit_load_metadata);
  m.def("commit_save_metadata", &commit_save_metadata);
  m.def("release_metadata", &release_metadata);
}
