/*
 *  Copyright (c), 2020, EPFL - Blue Brain Project
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#pragma once

#include <Eigen/Eigen>

namespace HighFive {

namespace details {

template<typename T, int M, int N>
struct data_converter<Eigen::Matrix<T, M, N>> {
    using value_type = Eigen::Matrix<T, M, N>;
    using inner_type = data_converter<T>;
    using dataspace_type = typename data_converter<T>::dataspace_type;

    inline data_converter(const std::vector<size_t>& dims)
    : _dims(dims)
    , _number_of_element(get_number_of_elements(dims))
    , _inner_converter(std::vector<size_t>(dims.begin() + 2, dims.end()))
    {
        if (dims.size() < 2) {
            throw std::string("Invalid number of dimensions");
        }
    }

    void allocate(value_type& val) {
        if (_dims[0] != static_cast<size_t>(val.rows())
         && _dims[1] != static_cast<size_t>(val.cols())) {
          val.resize(static_cast<typename value_type::Index>(_dims[0]),
                     static_cast<typename value_type::Index>(_dims[1]));
        }
    }

    static std::vector<size_t> get_size(const value_type& val) {
        return std::vector<size_t>{static_cast<size_t>(val.rows()), static_cast<size_t>(val.cols())};
    }

    dataspace_type* get_pointer(value_type& val) {
        return val.data();
    }

    const dataspace_type* get_pointer(const value_type& val) const {
        return val.data();
    }

    inline void process_result(value_type& vec, const dataspace_type* data) {
        for (unsigned int i = 0; i < vec.rows(); ++i) {
            for (unsigned int j = 0; j < vec.cols(); ++j) {
                vec(i, j) = data[i * vec.cols() + j];
            }
        }
    }

    inline void preprocess_result(const value_type& vec, dataspace_type* data) const {
        for (unsigned int i = 0; i < vec.rows(); ++i) {
            for (unsigned int j = 0; j < vec.cols(); ++j) {
                data[i * vec.cols() + j] = vec(i, j);
            }
        }
    }

    std::vector<size_t> _dims;
    size_t _number_of_element;
    inner_type _inner_converter;
};

template <typename S, int M, int N>
struct h5_continuous<Eigen::Matrix<S, M, N>> :
    std::true_type {};

}  // namespace details

}  // namespace HighFive
