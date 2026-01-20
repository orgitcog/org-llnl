// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#ifndef _KROWKEE_UTIL_COMPACT_ITERABLE_MAP_HPP
#define _KROWKEE_UTIL_COMPACT_ITERABLE_MAP_HPP

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace krowkee {
namespace util {

template <typename KeyType, typename ValueType>
using pair_vector_t = std::vector<std::pair<const KeyType, ValueType>>;

/**
 * Space-efficient map, designed to be a drop-in replacement for std::map.
 *
 * Core data structures are an archive std::vector and a small std::map
 * instance. Code is based heavily on an archived C++ User's Group article [0],
 * whose source code appears to be lost to time. There appear to be similar
 * implementations, e.g. from Apple [1], a hashmap implementation [2], and most
 * likely also in Boost, but there was no standalone, simple, open-source
 * solution to be had.
 *
 * References:
 *
 * [0]
 * https://www.drdobbs.com/space-efficient-sets-and-maps/184401668?queryText=carrato
 * [1]
 * https://developer.apple.com/documentation/swift/sequence/2950916-compactmap
 * [2] https://github.com/greg7mdp/sparsepp
 * [4] https://gist.github.com/jeetsukumaran/307264
 */
template <typename KeyType, typename ValueType,
          template <typename, typename> class MapType = std::map>
class compact_iterable_map {
 public:
  // Key needs to be const to agree with runtime map iterator typing.
  typedef std::pair<const KeyType, ValueType> pair_t;
  // typedef std::vector<pair_t> vec_t;
  typedef pair_vector_t<KeyType, ValueType> vec_t;
  typedef typename vec_t::iterator          vec_iter_t;
  // typedef std::map<KeyType, ValueType> map_t;
  // typedef MapType<KeyType, ValueType> map_t;
  typedef MapType<KeyType, ValueType>              map_t;
  typedef typename map_t::iterator                 map_iter_t;
  typedef compact_iterable_map<KeyType, ValueType> cm_t;

 public:
  /**
   * Iterator class for compact_iterable_map. Provides a unified interface for
   * STL algorithms that is agnostic to data structure details. Based in part on
   * [4].
   */
  class iterator {
   private:
    vec_iter_t _archive_iter;
    map_iter_t _dynamic_iter;
    bool       _archive_is_current;
    cm_t      *_ptr;

   public:
    iterator(cm_t *ptr)
        : _ptr(ptr),
          _archive_iter(std::begin(_ptr->_archive_map)),
          _dynamic_iter(std::begin(_ptr->_dynamic_map)) {
      _set_current_ascending();
    }
    iterator(cm_t *ptr, const vec_iter_t &axv_iter, const map_iter_t &dyn_iter,
             const bool archive_is_current)
        : _ptr(ptr),
          _archive_iter(axv_iter),
          _dynamic_iter(dyn_iter),
          _archive_is_current(archive_is_current) {}

    /**
     * Might need to check that we are not at std::end() for either.
     */
    iterator operator++() {
      if (_archive_is_current == true) {
        if (_archive_iter != std::end(_ptr->_archive_map)) {
          _archive_iter++;
          // _set_current_ascending();
        } else {  // we are at the end of the _archive_map
          if (_dynamic_iter != std::end(_ptr->_dynamic_map)) {
            _dynamic_iter++;
          } else {
            // _archive_is_current = _ptr->_archive_is_max;
          }
        }
      } else {
        if (_dynamic_iter != std::end(_ptr->_dynamic_map)) {
          _dynamic_iter++;
        } else {
          if (_archive_iter != std::end(_ptr->_archive_map)) {
            _archive_iter++;
          } else {
            // _archive_is_current = _ptr->_archive_is_max;
            // std::cout << "gets here" << std::endl;
          }
        }
      }
      _set_current_ascending();
      return *this;
    }

    /**
     * Probably need to check that we are not at std::begin() for either?
     * Ignoring for now.
     */
    iterator operator--() {
      const bool axv_is_begin(_archive_iter == std::begin(_ptr->_archive_map));
      const bool dyn_is_begin(_dynamic_iter == std::begin(_ptr->_dynamic_map));
      const bool axv_is_end(_archive_iter == std::end(_ptr->_archive_map));
      const bool dyn_is_end(_dynamic_iter == std::end(_ptr->_dynamic_map));
      if (axv_is_begin && dyn_is_begin) {  // both iterators are at beginning
        if (axv_is_end && axv_is_end) {    // both empty
          _archive_is_current = false;
        } else if (axv_is_end) {  // axv is empty
          _archive_is_current = false;
        } else if (dyn_is_end) {  // dyn is empy
          _archive_is_current = true;
        } else {                      // both at first element.
          if (_archive_is_current) {  // axv is current
            if (_archive_iter->first > _dynamic_iter->first) {  // not at begin
              _archive_is_current = false;
            } else {  // at begin
            }
          } else {  // dyn is current
            if (_archive_iter->first < _dynamic_iter->first) {  // not at begin
              _archive_is_current = true;
            } else {  // at begin
            }
          }
        }
      } else if (axv_is_begin) {  // dyn is not at rend
        if (axv_is_end) {         // axv is empty
          _dynamic_iter--;
          _archive_is_current == false;
        } else {                              // axv is at first element
          if (_archive_is_current == true) {  // axv is current
            _archive_is_current == false;     // "decrement" axv
          } else {                            // axv is not current
            if (_archive_iter->first < _dynamic_iter->first) {
              _dynamic_iter--;
              _archive_is_current = _archive_iter->first > _dynamic_iter->first;
            } else {  // axv is at rend.
              _dynamic_iter--;
            }
          }
          // if (_archive_is_current == false &&
          //     _archive_iter->first < _dynamic_iter->first) {
          //   _archive_iter--;
          //   _archive_is_current == true;
          // }
        }

      } else if (dyn_is_begin) {  // axv is not at rend
        if (dyn_is_end) {         // dyn is empty
          _archive_iter--;
          _archive_is_current == true;
        } else {                               // dyn is at first element
          if (_archive_is_current == false) {  // dyn is current
            _archive_is_current == true;       // "decrement" dyn
          } else {                             // dyn is not current
            if (_archive_iter->first >
                _dynamic_iter->first) {  // dyn not at rend
              _archive_iter--;
              _archive_is_current = _archive_iter->first > _dynamic_iter->first;
            } else {  // dyn is at rend.
              _archive_iter--;
            }
          }
          // if (_archive_is_current == false &&
          //     _archive_iter->first < _dynamic_iter->first) {
          //   _archive_iter--;
          //   _archive_is_current == true;
          // }
        }
      } else {  // guaranteed that both containers are not at rend.
        _archive_iter--;
        _dynamic_iter--;
        _archive_is_current = _archive_iter->first > _dynamic_iter->first;
        if (_archive_is_current) {
          _dynamic_iter++;
        } else {
          _archive_iter++;
        }
      }

      // bool axv_is_end(_archive_iter == std::end(_ptr->_archive_map));
      // bool dyn_is_end(_dynamic_iter == std::end(_ptr->_dynamic_map));
      // if (axv_is_end == true && dyn_is_end == true) {
      //   _archive_iter--;
      //   _dynamic_iter--;
      //   _set_current_descending();
      // } else if (axv_is_end == true) {
      //   _archive_iter--;
      //   _set_current_descending();
      //   if (_archive_is_current == false) { _dynamic_iter--; }
      // } else if (dyn_is_end == true) {
      //   _dynamic_iter--;
      //   _set_current_descending();
      //   if (_archive_is_current == true) { _archive_iter--; }
      // } else {
      //   _set_current_descending();
      //   if (_archive_is_current == true) {
      //     _archive_iter--;
      //   } else {
      //     _dynamic_iter--;
      //   }
      // }
      // _set_current_descending();
      // if (dyn_is_end == true) { _dynamic_iter--; }
      // _set_current_descending();
      // if (_archive_is_current == true) {
      //   _archive_iter--;
      // } else {
      //   _dynamic_iter--;
      // }
      return *this;
    }

    pair_t *operator->() {
      if (_archive_is_current == true) {
        return &(*_archive_iter);
      } else {
        return &(*_dynamic_iter);
      }
    }

    pair_t &operator*() {
      if (_archive_is_current == true) {
        return *_archive_iter;
      } else {
        return *_dynamic_iter;
      }
    }

    friend bool operator==(const iterator &lhs, const iterator &rhs) {
      return lhs._ptr == rhs._ptr && lhs._archive_iter == rhs._archive_iter &&
             lhs._dynamic_iter == rhs._dynamic_iter &&
             lhs._archive_is_current == rhs._archive_is_current;
    }
    friend bool operator!=(const iterator &lhs, const iterator &rhs) {
      return !(lhs == rhs);
    }

    friend std::ostream &operator<<(std::ostream &os, const iterator &iter) {
      os << *(iter._archive_iter) << ", " << *(iter._dynamic_iter) << ", "
         << ((iter._archive_is_current == true) ? "axv" : "dyn") << " current";
      return os;
    }

   private:
    /**
     * Might need more complex logic to check begin/end conditions.
     */
    void _set_current_ascending() {
      const bool axv_is_end(_archive_iter == std::end(_ptr->_archive_map));
      const bool dyn_is_end(_dynamic_iter == std::end(_ptr->_dynamic_map));
      if (axv_is_end && dyn_is_end) {  // both at end
        _archive_is_current = _ptr->_archive_is_max();
      } else if (axv_is_end) {  // only archive at end
        _archive_is_current = false;
      } else if (dyn_is_end) {  // only dynamic at end
        _archive_is_current = true;
      } else {  // neither at end
        _archive_is_current = _archive_iter->first < _dynamic_iter->first;
      }
    }

    void _set_current_descending() {
      const bool axv_is_begin(_archive_iter == std::begin(_ptr->_archive_map));
      const bool dyn_is_begin(_dynamic_iter == std::begin(_ptr->_dynamic_map));

      const bool archive_bigger = _archive_iter->first > _dynamic_iter->first;

      if (axv_is_begin && dyn_is_begin) {
      }
    }
  };

  typedef typename cm_t::iterator iter_t;
  // typedef cm_t::const_iterator const_iter_t;

 protected:
  std::vector<bool> _deleted;
  vec_t             _archive_map;
  map_t             _dynamic_map;
  std::size_t       _compaction_threshold;

  struct compare_first_f {
    bool operator()(const pair_t &lhs, const pair_t &rhs) const {
      return lhs.first < rhs.first;
    }
    bool operator()(const pair_t &lhs, const KeyType &key) const {
      return lhs.first < key;
    }
    bool operator()(const KeyType &key, const pair_t &rhs) const {
      return key < rhs.first;
    }
  };

  compare_first_f compare_first;

 public:
  compact_iterable_map(std::size_t compaction_threshold)
      : _compaction_threshold(compaction_threshold) {
    _archive_map.reserve(_compaction_threshold);
    _deleted.reserve(_compaction_threshold);
  }

  compact_iterable_map(const cm_t &rhs) {
    _archive_map.resize(rhs._archive_map.size());
    _deleted.resize(rhs._deleted.size());
    std::copy(std::begin(rhs._archive_map), std::end(rhs._archive_map),
              std::begin(_archive_map));
    std::copy(std::begin(rhs._dynamic_map), std::end(rhs._dynamic_map),
              std::begin(_dynamic_map));
    std::copy(std::begin(rhs._deleted), std::end(rhs._deleted),
              std::begin(_deleted));
    _compaction_threshold = rhs._compaction_threshold;
  }

  compact_iterable_map() {}

  compact_iterable_map(cm_t &&rhs) : cm_t() { std::swap(*this, rhs); }

  //////////////////////////////////////////////////////////////////////////////
  // Getters
  //////////////////////////////////////////////////////////////////////////////

  inline std::size_t size() const {
    return _archive_map.size() + _dynamic_map.size();
  }

  inline bool is_compact() const { return _dynamic_map.size() == 0; }

  constexpr std::size_t threshold() const { return _compaction_threshold; }

  //////////////////////////////////////////////////////////////////////////////
  // Compaction
  //////////////////////////////////////////////////////////////////////////////

  /**
   * Sort _dynamic_map into _archive_map, clearing deleted items along the way.
   *
   * @note[BWP] right now this does two copies, which is suboptimal. Should look
   * into ways to improve performance.
   */
  void compactify() {
    // copy nondeleted elements
    vec_t tmp1;
    tmp1.reserve(_archive_map.size());
    for (std::size_t i(0); i < _archive_map.size(); ++i) {
      if (_deleted[i] == false) {
        tmp1.push_back(_archive_map[i]);
      }
    }
    // perform union
    vec_t tmp2;
    tmp2.reserve(tmp2.size() + _dynamic_map.size());
    std::set_union(std::begin(tmp1), std::end(tmp1), std::begin(_dynamic_map),
                   std::end(_dynamic_map), std::back_inserter(tmp2),
                   compare_first);
    _dynamic_map.clear();
    std::swap(_archive_map, tmp2);
    _deleted.resize(_archive_map.size());
    std::fill(std::begin(_deleted), std::end(_deleted), false);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Insert
  //////////////////////////////////////////////////////////////////////////////

  std::pair<iter_t, bool> insert(const pair_t &pair) {
    // {  // search within _dynamic_map
    auto it_dyn(_dynamic_map.find(pair.first));
    if (it_dyn != std::end(_dynamic_map)) {
      auto axv_lb =
          std::lower_bound(std::begin(_archive_map), std::end(_archive_map),
                           pair, compare_first);
      return {iter_t{this, axv_lb, it_dyn, false}, false};
    }
    return try_insert_archive(pair);
  }

  // ValueType &operator[](const KeyType &key) {
  //   auto dyn_iter(_dynamic_map.find(key));
  //   if (dyn_iter != std::end(_dynamic_map)) { return dyn_iter->second; }

  //   auto [axv_iter, axv_code] = archive_find(key);
  //   return axv_iter->second;
  // }

  //////////////////////////////////////////////////////////////////////////////
  // Find
  //////////////////////////////////////////////////////////////////////////////

  iter_t find(const KeyType &key) {
    auto dyn_iter(_dynamic_map.find(key));
    if (dyn_iter != std::end(_dynamic_map)) {
      auto axv_lb = std::lower_bound(
          std::begin(_archive_map), std::end(_archive_map), key, compare_first);
      return {this, axv_lb, dyn_iter, false};
    }

    auto [axv_lb, axv_code] = archive_find(key);
    if (axv_code == archive_code_t::present) {
      auto dyn_lb = std::lower_bound(
          std::begin(_dynamic_map), std::end(_dynamic_map), key, compare_first);
      return {this, axv_lb, dyn_lb, true};
    }

    return end();
  }

  //////////////////////////////////////////////////////////////////////////////
  // Iterators
  //////////////////////////////////////////////////////////////////////////////

  constexpr iter_t begin() {
    return {this, std::begin(_archive_map), std::begin(_dynamic_map),
            _archive_is_min()};
  }

  constexpr iter_t end() {
    return {this, std::end(_archive_map), std::end(_dynamic_map),
            _archive_is_max()};
  }

  //////////////////////////////////////////////////////////////////////////////
  // Equality Operators
  //////////////////////////////////////////////////////////////////////////////

  // friend bool operator==(const cm_t &lhs, const cm_t &rhs) {
  //   return lhs._ptr == rhs._ptr && lhs._archive_iter == rhs._archive_iter &&
  //          lhs._dynamic_iter == rhs._dynamic_iter &&
  //          lhs._archive_is_current == rhs._archive_is_current;
  // }

  //////////////////////////////////////////////////////////////////////////////
  // I/O Operators
  //////////////////////////////////////////////////////////////////////////////

  std::string print_state() {
    std::stringstream ss{};
    ss << "axv (" << _archive_map.size() << "): ";
    for (auto iter(std::begin(_archive_map)); iter != std::end(_archive_map);
         ++iter) {
      ss << "(" << iter->first << "," << iter->second << ") ";
    }
    ss << '\n';
    ss << "dyn (" << _dynamic_map.size() << "): ";
    for (auto iter(std::begin(_dynamic_map)); iter != std::end(_dynamic_map);
         ++iter) {
      ss << "(" << iter->first << "," << iter->second << ") ";
    }
    ss << '\n';
    ss << "cmp: ";
    ss << _compaction_threshold;
    return ss.str();
  }

 private:
  enum class archive_code_t : std::uint8_t { present, deleted, absent };

  enum class dynamic_code_t : std::uint8_t { success, failure, compaction };

  inline std::pair<vec_iter_t, archive_code_t> archive_find(
      const KeyType &key) {
    auto axv_lb = std::lower_bound(std::begin(_archive_map),
                                   std::end(_archive_map), key, compare_first);
    bool is_present =
        (axv_lb != std::end(_archive_map) && axv_lb->first == key);
    if (is_present == false) {
      return {axv_lb, archive_code_t::absent};
    } else {
      int axv_offset = axv_lb - std::begin(_archive_map);
      if (_deleted[axv_offset] == true) {
        return {axv_lb, archive_code_t::deleted};
      } else {
        return {axv_lb, archive_code_t::present};
      }
    }
  }

  inline std::pair<map_iter_t, dynamic_code_t> dynamic_insert(
      const pair_t &pair) {
    auto [dyn_iter, success] = _dynamic_map.insert(pair);
    // check if we need to compactify
    if (_dynamic_map.size() == _compaction_threshold) {
      compactify();
      return {dyn_iter, dynamic_code_t::compaction};
    } else {
      return {dyn_iter, (success == true) ? dynamic_code_t::success
                                          : dynamic_code_t::failure};
    }
  }

  inline std::pair<iter_t, bool> try_insert_archive(const pair_t &pair) {
    auto [axv_lb, axv_code] = archive_find(pair.first);

    if (axv_code == archive_code_t::present) {
      auto dyn_lb =
          std::lower_bound(std::begin(_dynamic_map), std::end(_dynamic_map),
                           pair, compare_first);
      return {iter_t{this, axv_lb, dyn_lb, true}, false};
    } else if (axv_code == archive_code_t::deleted) {
      auto dyn_lb =
          std::lower_bound(std::begin(_dynamic_map), std::end(_dynamic_map),
                           pair, compare_first);
      axv_lb->second       = pair.second;
      int axv_offset       = axv_lb - std::begin(_archive_map);
      _deleted[axv_offset] = false;
      return {iter_t{this, axv_lb, dyn_lb, true}, true};
    } else {  // axv_code == archive_code_t::absent
              // insert into _dynamic_map
      // auto [dyn_iter, dyn_code] = dynamic_insert(pair);
      // if (dyn_code == dynamic_code_t::compaction) {
      //   auto axv_lb_new =
      //       std::lower_bound(std::begin(_archive_map),
      //       std::end(_archive_map),
      //                        pair, compare_first);
      //   return {iter_t{this, axv_lb_new, std::begin(_dynamic_map), true},
      //   true};
      // } else {
      //   return {iter_t{this, axv_lb, dyn_iter, false},
      //           (dyn_code == dynamic_code_t::success) ? true : false};
      // }
      auto [dyn_iter, dyn_code] = _dynamic_map.insert(pair);
      // check if we need to compactify
      if (_dynamic_map.size() == _compaction_threshold) {
        compactify();
        auto axv_lb_new =
            std::lower_bound(std::begin(_archive_map), std::end(_archive_map),
                             pair, compare_first);
        return {iter_t{this, axv_lb_new, std::begin(_dynamic_map), true}, true};
      } else {
        return {iter_t{this, axv_lb, dyn_iter, false}, dyn_code};
      }
    }
  }

  constexpr bool _archive_is_min() {
    if (_archive_map.size() == 0) {
      return false;
    }
    if (_dynamic_map.size() == 0) {
      return true;
    }
    KeyType dyn_key = std::begin(_dynamic_map)->first;
    KeyType axv_key = std::begin(_archive_map)->first;
    return axv_key < dyn_key;
  }

  constexpr bool _archive_is_max() {
    if (_dynamic_map.size() == 0) {
      return true;
    }
    if (_archive_map.size() == 0) {
      return false;
    }
    KeyType dyn_key = (--std::end(_dynamic_map))->first;
    KeyType axv_key = (--std::end(_archive_map))->first;
    return dyn_key < axv_key;
  }
};

}  // namespace util
}  // namespace krowkee

#endif