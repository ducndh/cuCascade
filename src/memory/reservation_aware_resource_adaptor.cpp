/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/notification_channel.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace cucascade {
namespace memory {

using impl_type                    = detail::reservation_aware_resource_adaptor_impl;
using stream_ordered_tracker_state = impl_type::stream_ordered_tracker_state;
using device_reserved_arena        = impl_type::device_reserved_arena;

namespace {

struct stream_ordered_allocation_tracker : public impl_type::allocation_tracker_iface {
  mutable std::mutex mutex;
  std::unordered_map<cudaStream_t, std::unique_ptr<stream_ordered_tracker_state>> stream_stats_map;

  stream_ordered_allocation_tracker() = default;

  void reset_tracker_state(rmm::cuda_stream_view stream) override
  {
    std::lock_guard lock(mutex);
    auto it = stream_stats_map.find(stream.value());
    if (it == stream_stats_map.end()) { return; }
    stream_stats_map.erase(stream.value());
  }

  void assign_reservation_to_tracker(rmm::cuda_stream_view stream,
                                     std::unique_ptr<device_reserved_arena> arena,
                                     std::unique_ptr<reservation_limit_policy> policy,
                                     std::unique_ptr<oom_handling_policy> oom_policy) override
  {
    std::lock_guard lock(mutex);
    auto it = stream_stats_map.find(stream.value());
    if (it != stream_stats_map.end()) {
      throw rmm::logic_error("Stream already has reservation state set");
    }

    stream_stats_map[stream.value()] = std::make_unique<stream_ordered_tracker_state>(
      std::move(arena), std::move(policy), std::move(oom_policy));
  }

  stream_ordered_tracker_state* get_tracker_state(rmm::cuda_stream_view stream) override
  {
    std::lock_guard lock(mutex);
    auto it = stream_stats_map.find(stream.value());
    if (it == stream_stats_map.end()) { return nullptr; }
    return it->second.get();
  }

  const stream_ordered_tracker_state* get_tracker_state(rmm::cuda_stream_view stream) const override
  {
    std::lock_guard lock(mutex);
    auto it = stream_stats_map.find(stream.value());
    if (it == stream_stats_map.end()) { return nullptr; }
    return it->second.get();
  }
};

struct ptds_allocation_tracker : public impl_type::allocation_tracker_iface {
  static inline thread_local std::unique_ptr<stream_ordered_tracker_state> thread_reservation_state;

  ptds_allocation_tracker() = default;

  void reset_tracker_state([[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    if (thread_reservation_state) { thread_reservation_state.reset(); }
  }

  void assign_reservation_to_tracker([[maybe_unused]] rmm::cuda_stream_view stream,
                                     std::unique_ptr<device_reserved_arena> arena,
                                     std::unique_ptr<reservation_limit_policy> policy,
                                     std::unique_ptr<oom_handling_policy> oom_policy) override
  {
    if (thread_reservation_state) {
      throw rmm::logic_error("Thread already has reservation state set");
    }

    thread_reservation_state = std::make_unique<stream_ordered_tracker_state>(
      std::move(arena), std::move(policy), std::move(oom_policy));
  }

  stream_ordered_tracker_state* get_tracker_state(
    [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    return thread_reservation_state.get();
  }

  const stream_ordered_tracker_state* get_tracker_state(
    [[maybe_unused]] rmm::cuda_stream_view stream) const override
  {
    return thread_reservation_state.get();
  }
};

}  // namespace

stream_ordered_tracker_state::stream_ordered_tracker_state(
  std::unique_ptr<device_reserved_arena> arena,
  std::unique_ptr<reservation_limit_policy> res_policy,
  std::unique_ptr<oom_handling_policy> oom_policy)
  : memory_reservation(std::move(arena)),
    reservation_policy(std::move(res_policy)),
    oom_policy(std::move(oom_policy))
{
}

std::size_t impl_type::stream_ordered_tracker_state::check_reservation_and_handle_overflow(
  [[maybe_unused]] impl_type& adaptor, std::size_t allocation_size, rmm::cuda_stream_view stream)
{
  int64_t stream_tracking_size       = static_cast<int64_t>(allocation_size);
  std::size_t upstream_tracking_size = allocation_size;

  auto reservation_size = static_cast<int64_t>(memory_reservation->size());
  int64_t arena_before  = memory_reservation->allocated_bytes.load();
  auto [success, post_allocation_inc] =
    memory_reservation->allocated_bytes.try_add(stream_tracking_size, reservation_size);

  // [INSTRUMENTATION] OVER-BUDGET-ADMISSION: try_add succeeds because the arena
  // is already NEGATIVE (cross-stream phantoms drove it below 0).  If the arena
  // had been at its correct value (>= 0), this same try_add would have failed
  // and triggered reservation_policy->handle_over_reservation() — i.e. the
  // pipeline would have been forced to either grow its reservation or back off.
  // PR #100 does not touch this code path; per-reservation budget enforcement
  // is silently bypassed.
  if (success && arena_before < 0) {
    int64_t admitted_via_negative_arena =
      std::min(stream_tracking_size, -arena_before);  // bytes that would have hit the cap
    static std::atomic<int> count{0};
    static std::atomic<long long> total_overspend{0};
    total_overspend.fetch_add(admitted_via_negative_arena, std::memory_order_relaxed);
    int n = count.fetch_add(1, std::memory_order_relaxed);
    if (n < 50) {
      std::fprintf(stderr,
                   "[OVER-BUDGET-ADMISSION #%d] arena_before=%ld alloc_size=%ld "
                   "reservation_size=%ld -> try_add SUCCEEDED (would have FAILED "
                   "with correct arena=0); %ld bytes of this allocation overflow "
                   "the reservation budget; running_total_overspend=%lld\n",
                   n,
                   arena_before,
                   stream_tracking_size,
                   reservation_size,
                   admitted_via_negative_arena,
                   total_overspend.load(std::memory_order_relaxed));
      std::fflush(stderr);
    }
  }

  if (!success) {
    if (reservation_policy) {
      std::lock_guard lock(_arbitration_mutex);
      reservation_policy->handle_over_reservation(stream,
                                                  allocation_size,
                                                  static_cast<std::size_t>(post_allocation_inc),
                                                  memory_reservation.get());
    }
    post_allocation_inc = memory_reservation->allocated_bytes.add(stream_tracking_size);
  }
  memory_reservation->peak_allocated_bytes.update_peak(post_allocation_inc);

  int64_t pre_allocation_inc = post_allocation_inc - stream_tracking_size;
  if (post_allocation_inc < reservation_size) {
    upstream_tracking_size = 0UL;
  } else if (pre_allocation_inc < reservation_size) {
    upstream_tracking_size = static_cast<std::size_t>(post_allocation_inc - reservation_size);
  }

  // [INSTRUMENTATION] UNTRACKED-ALLOC: when an arena was previously driven
  // negative by a cross-stream dealloc, subsequent allocs through this
  // reservation slip through try_add (current is negative, plenty of headroom)
  // and end up with upstream_tracking_size=0 -> they allocate REAL GPU memory
  // without incrementing _total_allocated_bytes.  PR #100's mask in
  // memory_space.cpp does not protect against this — the global counter
  // simply UNDER-counts actual GPU residency.
  if (pre_allocation_inc < 0 && upstream_tracking_size < allocation_size) {
    static std::atomic<int> count{0};
    static std::atomic<long long> total_untracked{0};
    long long missed = static_cast<long long>(allocation_size - upstream_tracking_size);
    total_untracked.fetch_add(missed, std::memory_order_relaxed);
    int n = count.fetch_add(1, std::memory_order_relaxed);
    if (n < 50) {
      std::fprintf(
        stderr,
        "[UNTRACKED-ALLOC #%d] pre_arena=%ld post_arena=%ld reservation_size=%ld "
        "alloc_size=%zu upstream_tracking_size=%zu -> missed=%lld bytes "
        "(real GPU memory allocated but _total_allocated_bytes not incremented); "
        "running_total_untracked=%lld\n",
        n, pre_allocation_inc, post_allocation_inc, reservation_size,
        allocation_size, upstream_tracking_size, missed,
        total_untracked.load(std::memory_order_relaxed));
      std::fflush(stderr);
    }
  }

  return upstream_tracking_size;
}

impl_type::reservation_aware_resource_adaptor_impl(
  memory_space_id space_id,
  rmm::device_async_resource_ref upstream,
  std::size_t capacity,
  std::unique_ptr<reservation_limit_policy> default_reservation_policy,
  std::unique_ptr<oom_handling_policy> default_oom_policy,
  AllocationTrackingScope tracking_scope,
  cudaMemPool_t pool_handle)
  : _space_id(space_id),
    _upstream(std::move(upstream)),
    _pool_handle(pool_handle),
    _memory_limit(capacity),
    _capacity(capacity),
    _allocation_tracker([&]() -> std::unique_ptr<allocation_tracker_iface> {
      if (tracking_scope == AllocationTrackingScope::PER_STREAM) {
        return std::make_unique<stream_ordered_allocation_tracker>();
      } else {
        return std::make_unique<ptds_allocation_tracker>();
      }
    }()),
    _default_reservation_policy(default_reservation_policy
                                  ? std::move(default_reservation_policy)
                                  : make_default_reservation_limit_policy()),
    _default_oom_policy(default_oom_policy ? std::move(default_oom_policy)
                                           : make_default_oom_policy())
{
}

impl_type::reservation_aware_resource_adaptor_impl(
  memory_space_id space_id,
  rmm::device_async_resource_ref upstream,
  std::size_t memory_limit,
  std::size_t capacity,
  std::unique_ptr<reservation_limit_policy> default_reservation_policy,
  std::unique_ptr<oom_handling_policy> default_oom_policy,
  AllocationTrackingScope tracking_scope,
  cudaMemPool_t pool_handle)
  : _space_id(space_id),
    _upstream(std::move(upstream)),
    _pool_handle(pool_handle),
    _memory_limit(memory_limit),
    _capacity(capacity),
    _allocation_tracker([&]() -> std::unique_ptr<allocation_tracker_iface> {
      if (tracking_scope == AllocationTrackingScope::PER_STREAM) {
        return std::make_unique<stream_ordered_allocation_tracker>();
      } else {
        return std::make_unique<ptds_allocation_tracker>();
      }
    }()),
    _default_reservation_policy(default_reservation_policy
                                  ? std::move(default_reservation_policy)
                                  : make_default_reservation_limit_policy()),
    _default_oom_policy(default_oom_policy ? std::move(default_oom_policy)
                                           : make_default_oom_policy())
{
}

rmm::device_async_resource_ref impl_type::get_upstream_resource() const noexcept
{
  return _upstream;
}

std::size_t impl_type::get_available_memory() const noexcept
{
  auto current_bytes = _total_allocated_bytes.load();
  return _capacity > current_bytes ? _capacity - current_bytes : 0;
}

std::size_t impl_type::get_available_memory(rmm::cuda_stream_view stream) const noexcept
{
  auto upstream_available_memory = get_available_memory();
  if (auto* state = _allocation_tracker->get_tracker_state(stream); state) {
    upstream_available_memory += state->memory_reservation->get_available_memory();
  }
  return upstream_available_memory;
}

std::size_t impl_type::get_available_memory_print(rmm::cuda_stream_view stream) const noexcept
{
  auto upstream_available_memory = get_available_memory();
  if (auto* state = _allocation_tracker->get_tracker_state(stream); state) {
    upstream_available_memory += state->memory_reservation->get_available_memory();
  }
  return upstream_available_memory;
}

std::size_t impl_type::get_allocated_bytes(rmm::cuda_stream_view stream) const
{
  const auto* stats = _allocation_tracker->get_tracker_state(stream);
  return stats ? static_cast<std::size_t>(
                   std::max(int64_t{0}, stats->memory_reservation->allocated_bytes.load()))
               : 0;
}

std::size_t impl_type::get_peak_allocated_bytes(rmm::cuda_stream_view stream) const
{
  const auto* stats = _allocation_tracker->get_tracker_state(stream);
  return stats ? static_cast<std::size_t>(
                   std::max(int64_t{0}, stats->memory_reservation->peak_allocated_bytes.peak()))
               : 0;
}

std::size_t impl_type::get_total_allocated_bytes() const { return _total_allocated_bytes.load(); }

std::size_t impl_type::get_peak_total_allocated_bytes() const
{
  return _peak_total_allocated_bytes.peak();
}

void impl_type::reset_peak_allocated_bytes(rmm::cuda_stream_view stream)
{
  auto* stats = _allocation_tracker->get_tracker_state(stream);
  if (stats) { stats->memory_reservation->peak_allocated_bytes.reset(0); }
}

std::size_t impl_type::get_total_reserved_bytes() const { return _total_reserved_bytes.load(); }

bool impl_type::is_stream_tracked(rmm::cuda_stream_view stream) const
{
  return _allocation_tracker->get_tracker_state(stream) != nullptr;
}

bool impl_type::attach_reservation_to_tracker(
  rmm::cuda_stream_view stream,
  std::unique_ptr<reservation> reserved_bytes,
  std::unique_ptr<reservation_limit_policy> stream_reservation_policy,
  std::unique_ptr<oom_handling_policy> stream_oom_policy)
{
  auto* stats = _allocation_tracker->get_tracker_state(stream);
  if (stats) { return false; }

  if (!stream_reservation_policy) {
    stream_reservation_policy = make_default_reservation_limit_policy();
  }

  if (!stream_oom_policy) { stream_oom_policy = make_default_oom_policy(); }

  _allocation_tracker->assign_reservation_to_tracker(
    stream,
    std::unique_ptr<device_reserved_arena>(
      dynamic_cast<device_reserved_arena*>(reserved_bytes->_arena.release())),
    std::move(stream_reservation_policy),
    std::move(stream_oom_policy));

  return true;
}
void impl_type::reset_stream_reservation(rmm::cuda_stream_view stream)
{
  _allocation_tracker->reset_tracker_state(stream);
}

std::unique_ptr<reserved_arena> impl_type::reserve(std::size_t bytes,
                                                   std::unique_ptr<event_notifier> release_notifer)
{
  if (do_reserve(bytes, _memory_limit)) {
    _number_of_allocations.fetch_add(1);
    return std::make_unique<device_reserved_arena>(*this, bytes, std::move(release_notifer));
  }
  return nullptr;
}

std::unique_ptr<reserved_arena> impl_type::reserve_upto(
  std::size_t bytes, std::unique_ptr<event_notifier> release_notifer)
{
  auto reserved_size = do_reserve_upto(bytes, _memory_limit);
  _number_of_allocations.fetch_add(1);
  return std::make_unique<device_reserved_arena>(*this, reserved_size, std::move(release_notifer));
}

bool impl_type::grow_reservation_by(device_reserved_arena& arena, std::size_t bytes)
{
  if (do_reserve(bytes, _memory_limit)) {
    arena._size += static_cast<int64_t>(bytes);
    return true;
  }
  return false;
}

void impl_type::shrink_reservation_to_fit(device_reserved_arena& arena)
{
  auto current_bytes = std::max(int64_t{0}, arena.allocated_bytes.load());
  if (current_bytes < arena.size()) {
    auto reclaimed_bytes = std::exchange(arena._size, current_bytes) - current_bytes;
    _total_allocated_bytes.sub(static_cast<std::size_t>(reclaimed_bytes));
  }
}

std::size_t impl_type::get_active_reservation_count() const noexcept
{
  return _number_of_allocations.load();
}

void* impl_type::allocate(cuda::stream_ref stream,
                          std::size_t bytes,
                          [[maybe_unused]] std::size_t alignment)
{
  auto* reservation_state = _allocation_tracker->get_tracker_state(stream);
  if (reservation_state != nullptr) {
    return do_allocate_managed(bytes, reservation_state, stream);
  } else {
    return do_allocate_managed(bytes, stream);
  }
}

void* impl_type::do_allocate_managed(std::size_t bytes, rmm::cuda_stream_view stream)
{
  auto tracking_size = rmm::align_up(bytes, 256);
  try {
    return do_allocate_unmanaged(bytes, tracking_size, stream);
  } catch (...) {
    return _default_oom_policy->handle_oom(bytes,
                                           stream,
                                           std::current_exception(),
                                           std::bind(&impl_type::do_allocate_unmanaged,
                                                     this,
                                                     std::placeholders::_1,
                                                     tracking_size,
                                                     std::placeholders::_2));
  }
}

void* impl_type::do_allocate_managed(std::size_t bytes,
                                     stream_ordered_tracker_state* state,
                                     rmm::cuda_stream_view stream)
{
  auto padded_bytes  = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto tracking_size = state->check_reservation_and_handle_overflow(*this, padded_bytes, stream);
  try {
    return do_allocate_unmanaged(bytes, tracking_size, stream);
  } catch (...) {
    try {
      return state->oom_policy->handle_oom(bytes,
                                           stream,
                                           std::current_exception(),
                                           std::bind(&impl_type::do_allocate_unmanaged,
                                                     this,
                                                     std::placeholders::_1,
                                                     tracking_size,
                                                     std::placeholders::_2));
    } catch (...) {
      state->memory_reservation->allocated_bytes.sub(static_cast<int64_t>(padded_bytes));
      throw;
    }
  }
}

void* impl_type::do_allocate_unmanaged(std::size_t allocation_bytes,
                                       std::size_t tracking_bytes,
                                       rmm::cuda_stream_view stream)
{
  auto [success, post_allocation_size] = _total_allocated_bytes.try_add(tracking_bytes, _capacity);
  if (success) {
    _peak_total_allocated_bytes.update_peak(post_allocation_size);
    try {
      return _upstream.allocate(stream, allocation_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    } catch (std::exception& e) {
      _total_allocated_bytes.sub(tracking_bytes);
      throw cucascade_out_of_memory(e.what(),
                                    MemoryError::ALLOCATION_FAILED,
                                    allocation_bytes,
                                    post_allocation_size,
                                    _pool_handle);
    }
  } else {
    throw cucascade_out_of_memory("not enough capacity to allocate memory",
                                  MemoryError::POOL_EXHAUSTED,
                                  allocation_bytes,
                                  post_allocation_size,
                                  _pool_handle);
  }
}

void impl_type::deallocate(cuda::stream_ref stream,
                           void* ptr,
                           std::size_t bytes,
                           [[maybe_unused]] std::size_t alignment) noexcept
{
  auto tracking_bytes           = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto upstream_reclaimed_bytes = tracking_bytes;
  auto* reservation_state       = _allocation_tracker->get_tracker_state(stream);
  if (reservation_state != nullptr) {
    auto* reservation     = reservation_state->memory_reservation.get();
    auto reservation_size = static_cast<int64_t>(reservation->size());
    int64_t post_deallocation_size =
      reservation->allocated_bytes.sub(static_cast<int64_t>(tracking_bytes));
    int64_t pre_deallocation_size = post_deallocation_size + static_cast<int64_t>(tracking_bytes);
    if (pre_deallocation_size <= reservation_size) {
      // if it was made using the reserved space
      upstream_reclaimed_bytes = 0;
      // [INSTRUMENTATION] cross-stream dealloc: this stream's reservation never charged
      // for this buffer (the buffer was alloc'd on a different stream with no reservation
      // attached). The sub() above just drove arena.allocated_bytes negative.
      if (post_deallocation_size < 0) {
        static std::atomic<int> count{0};
        int n = count.fetch_add(1, std::memory_order_relaxed);
        if (n < 50) {
          std::fprintf(
            stderr,
            "[CROSS-STREAM-DEALLOC #%d] tracking=%zu arena.allocated_bytes %ld -> %ld "
            "(NEGATIVE; reservation_size=%ld) -> upstream_reclaimed=0 -> "
            "_total_allocated_bytes NOT decremented\n",
            n, tracking_bytes, pre_deallocation_size, post_deallocation_size, reservation_size);
          std::fflush(stderr);
        }
      }
    } else if (post_deallocation_size < reservation_size) {
      // if it was partially made using the reserved space
      upstream_reclaimed_bytes = static_cast<std::size_t>(pre_deallocation_size - reservation_size);
    }
  }
// Suppress false-positive null-dereference warnings from CCCL library code
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
  _upstream.deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
#pragma GCC diagnostic pop
  _total_allocated_bytes.sub(upstream_reclaimed_bytes);
}

bool impl_type::operator==(impl_type const& other) const noexcept
{
// Suppress false-positive null-dereference warnings from CCCL library code
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
  return _upstream == other.get_upstream_resource();
#pragma GCC diagnostic pop
}

bool impl_type::do_reserve(std::size_t size_bytes, std::size_t limit_bytes)
{
  auto [success, post_increase_bytes] = _total_allocated_bytes.try_add(size_bytes, limit_bytes);
  if (success) {
    _peak_total_allocated_bytes.update_peak(post_increase_bytes);
    _total_reserved_bytes.fetch_add(size_bytes);
  }
  return success;
}

std::size_t impl_type::do_reserve_upto(std::size_t size_bytes, std::size_t limit_bytes)
{
  auto post_increase_bytes = _total_allocated_bytes.add_bounded(size_bytes, limit_bytes);
  if (post_increase_bytes > 0) {
    _peak_total_allocated_bytes.update_peak(post_increase_bytes);
    _total_reserved_bytes.fetch_add(size_bytes);
  }
  return size_bytes;
}

void impl_type::do_release_reservation(device_reserved_arena* arena) noexcept
{
  if (!arena) return;

  int64_t allocation_size    = arena->allocated_bytes.load();
  int64_t arena_size         = arena->size();
  std::size_t released_bytes = 0;
  if (arena_size > allocation_size) {
    released_bytes = static_cast<std::size_t>(arena_size - allocation_size);
  }

  // [INSTRUMENTATION] inflated release: when allocation_size is negative (caused by the
  // cross-stream dealloc pattern), released_bytes = arena_size - allocation_size =
  // arena_size + |allocation_size|. Subtracted from _total_allocated_bytes, this drains
  // the global counter by MORE than the reservation's true contribution.
  if (allocation_size < 0) {
    static std::atomic<int> count{0};
    static std::atomic<int> wrap_count{0};
    auto cur_total =
      static_cast<long long>(_total_allocated_bytes.load(std::memory_order_relaxed));
    int n = count.fetch_add(1, std::memory_order_relaxed);
    bool would_wrap =
      cur_total >= 0 && static_cast<long long>(released_bytes) > cur_total;
    if (would_wrap) {
      int wn = wrap_count.fetch_add(1, std::memory_order_relaxed);
      std::fprintf(stderr,
                   "[!!! WRAP-OBSERVED #%d !!!] arena_size=%ld alloc_bytes=%ld "
                   "released_bytes=%zu cur_counter=%lld -> fetch_sub will WRAP "
                   "this counter to ~UINT64_MAX (PR #100 hides the symptom)\n",
                   wn, arena_size, allocation_size, released_bytes, cur_total);
      std::fflush(stderr);
    }
    if (n < 50 || would_wrap) {
      std::fprintf(stderr,
                   "[INFLATED-RELEASE #%d] arena_size=%ld alloc_bytes=%ld -> "
                   "released_bytes=%zu (over-drain by %ld bytes); "
                   "_total_allocated_bytes before sub = %lld\n",
                   n,
                   arena_size,
                   allocation_size,
                   released_bytes,
                   -allocation_size,
                   cur_total);
      std::fflush(stderr);
    }
  }

  _number_of_allocations.fetch_sub(1);
  _total_reserved_bytes.fetch_sub(static_cast<std::size_t>(std::max(int64_t{0}, arena_size)));
  // [INSTRUMENTATION] capture wrap as it happens — sub() returns the new value;
  // if it's huge (UINT64_MAX-region), the unsigned subtract underflowed.
  std::size_t post_sub = _total_allocated_bytes.sub(released_bytes);
  if (post_sub > (std::size_t{1} << 60) && released_bytes > 0) {
    static std::atomic<int> wrap_n{0};
    int wn = wrap_n.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr,
                 "[!!! WRAP-FIRED #%d !!!] released=%zu -> _total_allocated_bytes "
                 "WRAPPED to %zu. Every subsequent try_add(size, limit) in "
                 "do_reserve / do_allocate_unmanaged will fail because "
                 "current > limit. PR #100 silences should_downgrade_memory but "
                 "the engine is now bricked: queries fail with POOL_EXHAUSTED "
                 "while nvidia-smi shows free GPU memory.\n",
                 wn, released_bytes, post_sub);
    std::fflush(stderr);
  }
}

void impl_type::set_default_policy(std::unique_ptr<reservation_limit_policy> policy)
{
  _default_reservation_policy = std::move(policy);
}

const reservation_limit_policy& impl_type::get_default_reservation_policy() const
{
  return *_default_reservation_policy;
}

const oom_handling_policy& impl_type::get_default_oom_handling_policy() const
{
  return *_default_oom_policy;
}

reservation_aware_resource_adaptor::reservation_aware_resource_adaptor(
  memory_space_id space_id,
  rmm::device_async_resource_ref upstream,
  std::size_t capacity,
  std::unique_ptr<reservation_limit_policy> stream_reservation_policy,
  std::unique_ptr<oom_handling_policy> default_oom_policy,
  AllocationTrackingScope tracking_scope,
  cudaMemPool_t pool_handle)
  : shared_base(cuda::mr::make_shared_resource<impl_type>(space_id,
                                                          std::move(upstream),
                                                          capacity,
                                                          std::move(stream_reservation_policy),
                                                          std::move(default_oom_policy),
                                                          tracking_scope,
                                                          pool_handle))
{
}

reservation_aware_resource_adaptor::reservation_aware_resource_adaptor(
  memory_space_id space_id,
  rmm::device_async_resource_ref upstream,
  std::size_t memory_limit,
  std::size_t capacity,
  std::unique_ptr<reservation_limit_policy> stream_reservation_policy,
  std::unique_ptr<oom_handling_policy> default_oom_policy,
  AllocationTrackingScope tracking_scope,
  cudaMemPool_t pool_handle)
  : shared_base(cuda::mr::make_shared_resource<impl_type>(space_id,
                                                          std::move(upstream),
                                                          memory_limit,
                                                          capacity,
                                                          std::move(stream_reservation_policy),
                                                          std::move(default_oom_policy),
                                                          tracking_scope,
                                                          pool_handle))
{
}

rmm::device_async_resource_ref reservation_aware_resource_adaptor::get_upstream_resource()
  const noexcept
{
  return get().get_upstream_resource();
}

std::size_t reservation_aware_resource_adaptor::get_available_memory() const noexcept
{
  return get().get_available_memory();
}

std::size_t reservation_aware_resource_adaptor::get_available_memory(
  rmm::cuda_stream_view stream) const noexcept
{
  return get().get_available_memory(stream);
}

std::size_t reservation_aware_resource_adaptor::get_available_memory_print(
  rmm::cuda_stream_view stream) const noexcept
{
  return get().get_available_memory_print(stream);
}

std::size_t reservation_aware_resource_adaptor::get_allocated_bytes(
  rmm::cuda_stream_view stream) const
{
  return get().get_allocated_bytes(stream);
}

std::size_t reservation_aware_resource_adaptor::get_peak_allocated_bytes(
  rmm::cuda_stream_view stream) const
{
  return get().get_peak_allocated_bytes(stream);
}

std::size_t reservation_aware_resource_adaptor::get_total_allocated_bytes() const
{
  return get().get_total_allocated_bytes();
}

std::size_t reservation_aware_resource_adaptor::get_peak_total_allocated_bytes() const
{
  return get().get_peak_total_allocated_bytes();
}

void reservation_aware_resource_adaptor::reset_peak_allocated_bytes(rmm::cuda_stream_view stream)
{
  get().reset_peak_allocated_bytes(stream);
}

std::size_t reservation_aware_resource_adaptor::get_total_reserved_bytes() const
{
  return get().get_total_reserved_bytes();
}

bool reservation_aware_resource_adaptor::is_stream_tracked(rmm::cuda_stream_view stream) const
{
  return get().is_stream_tracked(stream);
}

std::unique_ptr<reserved_arena> reservation_aware_resource_adaptor::reserve(
  std::size_t bytes, std::unique_ptr<event_notifier> release_notifer)
{
  return get().reserve(bytes, std::move(release_notifer));
}

std::unique_ptr<reserved_arena> reservation_aware_resource_adaptor::reserve_upto(
  std::size_t bytes, std::unique_ptr<event_notifier> release_notifer)
{
  return get().reserve_upto(bytes, std::move(release_notifer));
}

std::size_t reservation_aware_resource_adaptor::get_active_reservation_count() const noexcept
{
  return get().get_active_reservation_count();
}

bool reservation_aware_resource_adaptor::attach_reservation_to_tracker(
  rmm::cuda_stream_view stream,
  std::unique_ptr<reservation> reserved_bytes,
  std::unique_ptr<reservation_limit_policy> stream_reservation_policy,
  std::unique_ptr<oom_handling_policy> stream_oom_policy)
{
  return get().attach_reservation_to_tracker(stream,
                                             std::move(reserved_bytes),
                                             std::move(stream_reservation_policy),
                                             std::move(stream_oom_policy));
}

void reservation_aware_resource_adaptor::reset_stream_reservation(rmm::cuda_stream_view stream)
{
  get().reset_stream_reservation(stream);
}

void reservation_aware_resource_adaptor::set_default_policy(
  std::unique_ptr<reservation_limit_policy> policy)
{
  get().set_default_policy(std::move(policy));
}

const reservation_limit_policy& reservation_aware_resource_adaptor::get_default_reservation_policy()
  const
{
  return get().get_default_reservation_policy();
}

const oom_handling_policy& reservation_aware_resource_adaptor::get_default_oom_handling_policy()
  const
{
  return get().get_default_oom_handling_policy();
}

}  // namespace memory
}  // namespace cucascade
