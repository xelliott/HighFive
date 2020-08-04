/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5UTILS_HPP
#define H5UTILS_HPP

// internal utilities functions
#include <algorithm>
#include <array>
#include <cstddef> // __GLIBCXX__
#include <exception>
#include <string>
#include <type_traits>
#include <vector>

#ifdef H5_USE_BOOST
# include <boost/multi_array.hpp>
# include <boost/numeric/ublas/matrix.hpp>
#endif
#ifdef H5_USE_EIGEN
# include <Eigen/Eigen>
#endif

#include <H5public.h>

namespace HighFive {

namespace details {

// check if the type is a basic C-Array
template <typename>
struct is_c_array {
    static const bool value = false;
};

template <typename T>
struct is_c_array<T*> {
    static const bool value = true;
};

template <typename T, std::size_t N>
struct is_c_array<T[N]> {
    static const bool value = true;
};


// converter function for hsize_t -> size_t when hsize_t != size_t
template <typename Size>
inline std::vector<std::size_t> to_vector_size_t(const std::vector<Size>& vec) {
    static_assert(std::is_same<Size, std::size_t>::value == false,
                  " hsize_t != size_t mandatory here");
    std::vector<size_t> res(vec.size());
    std::transform(vec.cbegin(), vec.cend(), res.begin(), [](Size e) {
        return static_cast<size_t>(e);
    });
    return res;
}

// converter function for hsize_t -> size_t when size_t == hsize_t
inline std::vector<std::size_t> to_vector_size_t(const std::vector<std::size_t>& vec) {
    return vec;
}

}  // namespace details
}  // namespace HighFive

#endif // H5UTILS_HPP
