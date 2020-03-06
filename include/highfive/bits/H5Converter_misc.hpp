/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5CONVERTER_MISC_HPP
#define H5CONVERTER_MISC_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>

#ifdef H5_USE_BOOST
// starting Boost 1.64, serialization header must come before ublas
#include <boost/serialization/vector.hpp>
#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#endif

#include <H5Dpublic.h>
#include <H5Ppublic.h>

#include "../H5Reference.hpp"
#include "H5Utils.hpp"

namespace HighFive {

namespace details {
template <typename T>
struct h5_pod :
    std::is_trivial<T> {};

template <>
struct h5_pod<bool> :
    std::false_type {};

// contiguous pair of T
template <typename T>
struct h5_pod<std::complex<T>> :
    std::true_type {};

template <typename T>
struct h5_continuous :
    std::false_type {};

template <typename S>
struct h5_continuous<std::vector<S>> :
    std::integral_constant<bool, h5_pod<S>::value> {};

template <typename T, size_t N>
struct h5_continuous<T[N]> :
    std::integral_constant<bool, h5_pod<T>::value> {};

template <typename S, size_t N>
struct h5_continuous<std::array<S, N>> :
    std::integral_constant<bool, h5_pod<S>::value> {};

template <typename T, size_t Dims>
struct h5_continuous<boost::multi_array<T, Dims>> :
    std::integral_constant<bool, h5_pod<T>::value> {};

template <typename T>
struct h5_continuous<boost::numeric::ublas::matrix<T>> :
    std::integral_constant<bool, h5_pod<T>::value> {};

// template <typename T, int M, int N>
// struct h5_continuous<Eigen::Matrix<T, M, N>> :
//     std::true_type {};

template <typename T>
struct h5_non_continuous :
    std::integral_constant< bool, !h5_continuous<T>::value> {};

size_t get_number_of_elements(const std::vector<size_t>& dims) {
    return std::accumulate(dims.begin(), dims.end(), size_t{1u}, std::multiplies<size_t>());
}

template <typename T>
struct data_converter {
    using value_type = T;
    using inner_type = T;
    using dataspace_type = T;

    inline data_converter(const std::vector<size_t>& dims)
      : _number_of_element(get_number_of_elements(dims))
    {
        if (dims.size() > 1 || _number_of_element > 1) {
            throw std::string("Invalid number of elements");
        }
    }

    void allocate(value_type&) {
        // pass
    }

    static std::vector<size_t> get_size(const value_type& val) {
        return std::vector<size_t>{1};
    }

    dataspace_type* get_pointer(value_type& val) {
        return &val;
    }

    const dataspace_type* get_pointer(const value_type& val) const {
        return &val;
    }

    inline void process_result(value_type& scalar, const dataspace_type* data) {
        scalar = *data;
    }

    inline void preprocess_result(const value_type& scalar, dataspace_type* data) {
        *data = scalar;
    }

    size_t _number_of_element;
};

// std::string is an array of char
template<>
struct data_converter<std::string> {
    using value_type = std::string;
    using dataspace_type = char;

    inline data_converter(const std::vector<size_t>& dims)
      : _dims(dims)
      , _number_of_element(get_number_of_elements(dims))
    {
        if (dims.size() > 1) {
            throw std::string("Invalid number of elements");
        }
    }

    void allocate(value_type&) {
        // pass
    }

    static std::vector<size_t> get_size(const value_type& val) {
        return std::vector<size_t>{val.size()};
    }

    dataspace_type* get_pointer(value_type& val) {
        // .data() return a const pointer until c++17
        throw std::string("Invalid get_pointer on std::string");
    }

    const dataspace_type* get_pointer(const value_type& val) const {
        return val.data();
    }

    inline void process_result(value_type& scalar, const dataspace_type* data) {
        scalar = std::string(data, _dims[0]);
    }

    inline void preprocess_result(const value_type& scalar, dataspace_type* data) {
        scalar.copy(data, _dims[0]);
    }

    std::vector<size_t> _dims;
    size_t _number_of_element;
};

template <typename T>
struct data_converter<std::vector<T>> {
    using value_type = std::vector<T>;
    using inner_type = data_converter<T>;
    using dataspace_type = typename data_converter<T>::dataspace_type;

    inline data_converter(const std::vector<size_t>& dims)
      : _dims(dims)
      , _number_of_element(get_number_of_elements(dims))
      , _inner_converter(std::vector<size_t>(dims.begin() + 1, dims.end()))
    { };

    void allocate(value_type& val) {
        val.resize(_dims[0]);
        for (auto& v: val) {
            _inner_converter.allocate(v);
        }
    }

    static std::vector<size_t> get_size(const value_type& val) {
        auto ret = inner_type::get_size(val[0]);
        ret.insert(ret.begin(), val.size());
        return ret;
    }

    dataspace_type* get_pointer(value_type& val) {
        return val.data();
    }

    const dataspace_type* get_pointer(const value_type& val) const {
        return val.data();
    }

    // Internal function that take data and put it in vec
    inline void process_result(value_type& vec, const dataspace_type* data) {
        for (unsigned int i = 0; i < _dims[0]; ++i) {
            _inner_converter.process_result(vec[i], data + _inner_converter._number_of_element * i); 
        }
    }

    inline void preprocess_result(const value_type& scalar, dataspace_type* data) {
         for (size_t i = 0; i < scalar.size(); ++i) {
             _inner_converter.preprocess_result(scalar[i], data + _inner_converter._number_of_element * i);
         }
    }

    std::vector<size_t> _dims;
    size_t _number_of_element;
    inner_type _inner_converter;

};

template <typename T, size_t N>
struct data_converter<T[N]> {
    using value_type = T[N];
    using inner_type = data_converter<T>;
    using dataspace_type = typename data_converter<T>::dataspace_type;

    inline data_converter(const std::vector<size_t>& dims)
      : _dims(dims)
      , _number_of_element(get_number_of_elements(dims))
      , _inner_converter(std::vector<size_t>(dims.begin() + 1, dims.end()))
    { };

    void allocate(value_type& val) {
        // pass
    }

    static std::vector<size_t> get_size(const value_type& val) {
        auto ret = inner_type::get_size(val[0]);
        ret.insert(ret.begin(), N);
        return ret;
    }

    dataspace_type* get_pointer(value_type& val) {
        return val;
    }

    const dataspace_type* get_pointer(const value_type& val) const {
        return val;
    }

    // Internal function that take data and put it in vec
    inline void process_result(value_type& vec, const dataspace_type* data) {
        for (unsigned int i = 0; i < _dims[0]; ++i) {
            _inner_converter.process_result(vec[i], data + _inner_converter._number_of_element * i); 
        }
    }

    inline void preprocess_result(const value_type& scalar, dataspace_type* data) {
         for (size_t i = 0; i < scalar.size(); ++i) {
             _inner_converter.preprocess_result(scalar[i], data + _inner_converter._number_of_element * i);
         }
    }

    std::vector<size_t> _dims;
    size_t _number_of_element;
    inner_type _inner_converter;
};

template <typename T, size_t N>
struct data_converter<std::array<T, N>> {
    using value_type = std::array<T, N>;
    using inner_type = data_converter<T>;
    using dataspace_type = typename data_converter<T>::dataspace_type;

    inline data_converter(const std::vector<size_t>& dims)
      : _dims(dims)
      , _number_of_element(get_number_of_elements(dims))
      , _inner_converter(std::vector<size_t>(dims.begin() + 1, dims.end()))
    { };

    void allocate(value_type& val) {
        for(auto& v: val) {
            _inner_converter.allocate(v);
        }
    }

    static std::vector<size_t> get_size(const value_type& val) {
        auto ret = inner_type::get_size(val[0]);
        ret.insert(ret.begin(), N);
        return ret;
    }

    dataspace_type* get_pointer(value_type& val) {
        return val.data();
    }

    const dataspace_type* get_pointer(const value_type& val) const {
        return val.data();
    }

    // Internal function that take data and put it in vec
    inline void process_result(value_type& vec, const dataspace_type* data) {
        for (unsigned int i = 0; i < _dims[0]; ++i) {
            _inner_converter.process_result(vec[i], data + _inner_converter._number_of_element * i); 
        }
    }

    inline void preprocess_result(const value_type& scalar, dataspace_type* data) {
         for (int i = 0; i < scalar.size(); ++i) {
             _inner_converter.preprocess_result(scalar[i], data + _inner_converter._number_of_element * i);
         }
    }

    std::vector<size_t> _dims;
    size_t _number_of_element;
    inner_type _inner_converter;

};

#ifdef H5_USE_BOOST
// apply conversion to boost multi array
template <typename T, std::size_t Dims>
struct data_converter<boost::multi_array<T, Dims>> {
    using value_type = boost::multi_array<T, Dims>;
    using inner_type = data_converter<T>;
    using dataspace_type = typename data_converter<T>::dataspace_type;

    inline data_converter(const std::vector<size_t>& dims)
    : _dims(dims)
    , _number_of_element(get_number_of_elements(dims))
    , _inner_converter(std::vector<size_t>(dims.begin() + Dims, dims.end()))
    {
        if (dims.size() < Dims) {
            throw std::string("Invalid number of dimensions");
        }
    }

    void allocate(value_type& val) {
        if (std::equal(_dims.begin(), _dims.begin() + Dims, val.shape()) == false) {
            boost::array<typename value_type::index, Dims> ext;
            std::copy(_dims.begin(), _dims.begin() + Dims, ext.begin());
            val.resize(ext);
        }
    }

    static std::vector<size_t> get_size(const value_type& val) {
        auto ret = inner_type::get_size(val.data()[0]);
        for (unsigned i = Dims; i > 0; --i) {
            ret.insert(ret.begin(), val.shape()[i-1]);
        }
        return ret;
    }

    dataspace_type* get_pointer(value_type& val) {
        return val.data();
    }

    const dataspace_type* get_pointer(const value_type& val) const {
        return val.data();
    }

    inline void process_result(value_type& vec, const dataspace_type* data) {
        for (unsigned int i = 0; i < vec.num_elements(); ++i) {
            _inner_converter.process_result(vec.data()[i], data + i * _inner_converter._number_of_element);
        }
    }

    inline void preprocess_result(const value_type& scalar, dataspace_type* data) {
         for (int i = 0; i < scalar.num_elements(); ++i) {
             _inner_converter.preprocess_result(scalar.data()[i], data + _inner_converter._number_of_element * i);
         }
    }

    std::vector<size_t> _dims;
    size_t _number_of_element;
    inner_type _inner_converter;
};

template <typename T>
struct data_converter<boost::numeric::ublas::matrix<T>> {
    using value_type = boost::numeric::ublas::matrix<T>;
    using inner_type = data_converter<T>;
    using dataspace_type = typename data_converter<T>::dataspace_type;

    inline data_converter(const std::vector<size_t>& dims)
    : _dims(dims)
    , _number_of_element(get_number_of_elements(dims))
    , _inner_converter(std::vector<size_t>(dims.begin() + 2, dims.end()))
    {
        if (dims.size() < 2) {
            throw std::string("Invalid number of dimensions for boost UBLAS matrixes");
        }
    }

    void allocate(value_type& val) {
        std::array<size_t, 2> sizes = {{val.size1(), val.size2()}};
        if (std::equal(_dims.begin(), _dims.begin() + 2, sizes.begin()) == false) {
            val.resize(_dims[0], _dims[1], false);
            val(0, 0) = 0;
        }
    }

    static std::vector<size_t> get_size(const value_type& val) {
        auto ret = inner_type::get_size(val.data()[0]);
        ret.insert(ret.begin(), val.size2());
        ret.insert(ret.begin(), val.size1());
        return ret;
    }

    dataspace_type* get_pointer(value_type& val) {
        return &val(0, 0);
    }

    const dataspace_type* get_pointer(const value_type& val) const {
        return &val(0, 0);
    }

    inline void process_result(value_type& vec, const dataspace_type* data) {
        for (unsigned int i = 0; i < vec.size1() * vec.size2(); ++i) {
            _inner_converter.process_result(vec.data()[i], data + i * _inner_converter._number_of_element);
        }
    }

    inline void preprocess_result(const value_type& scalar, dataspace_type* data) {
         for (int i = 0; i < scalar.size1() * scalar.size2(); ++i) {
             _inner_converter.preprocess_result(scalar.data()[i], data + _inner_converter._number_of_element * i);
         }
    }

    std::vector<size_t> _dims;
    size_t _number_of_element;
    inner_type _inner_converter;
};
#endif
}  // namespace details

template <typename T, typename Enable = void>
class TransformRead;

template <typename T>
class TransformRead<T, typename std::enable_if<details::h5_non_continuous<T>::value>::type> {
  public:
    using dataspace_type = typename details::data_converter<T>::dataspace_type;
    using Conv = details::data_converter<T>;

    TransformRead(const std::vector<size_t>& dims)
        : _dims(dims)
        , _converter(_dims)
    {
#ifdef H5_ENABLE_ERROR_ON_COPY
#error You are using a non continuous data type and so data will be copied
#endif
    }

    dataspace_type* get_pointer() {
        _vec.resize(details::get_number_of_elements(_dims));
        return _vec.data();
    }

    const dataspace_type* get_pointer() const {
        _vec.resize(details::get_number_of_elements(_dims));
        return _vec.data();
    }

    T transform_read() {
        _converter.allocate(_data);
        _converter.process_result(_data, _vec.data());
        return _data;
    }

  private:
    // Continuous vector for data
    std::vector<dataspace_type> _vec;
    T _data;

    std::vector<size_t> _dims;
    Conv _converter;
};

template<typename T>
class TransformRead<T, typename std::enable_if<details::h5_continuous<T>::value>::type> {
  public:
    using dataspace_type = typename details::data_converter<T>::dataspace_type;
    using Conv = details::data_converter<T>;

    TransformRead(const std::vector<size_t>& dims)
        : _dims(dims)
        , _converter(_dims)
    {
    }

    dataspace_type* get_pointer() {
        _converter.allocate(_data);
        return _converter.get_pointer(_data);
    }

    T transform_read() {
        return _data;
    }

  private:
    T _data;
    std::vector<size_t> _dims;
    Conv _converter;
};

template <typename T, typename Enable = void>
class TransformWrite;

template <typename T>
class TransformWrite<T, typename std::enable_if<details::h5_non_continuous<T>::value>::type> {
  public:
    using dataspace_type = typename details::data_converter<T>::dataspace_type;
    using Conv = details::data_converter<T>;

    TransformWrite(const T& value)
        : _dims(Conv::get_size(value))
        , _converter(_dims)
        , _data(value)
    {
#ifdef H5_ENABLE_ERROR_ON_COPY
#error You are using a non continuous data type and so data will be copied
#endif
        _vec.resize(details::get_number_of_elements(_dims));
        _converter.preprocess_result(value, _vec.data());
    }

    const dataspace_type* get_pointer() const {
        return _vec.data();
    }

  private:
    std::vector<size_t> _dims;
    Conv _converter;
    const T& _data;
    std::vector<dataspace_type> _vec;
};

template <typename T>
class TransformWrite<T, typename std::enable_if<details::h5_continuous<T>::value>::type> {
  public:
    using dataspace_type = typename details::data_converter<T>::dataspace_type;
    using Conv = details::data_converter<T>;

    TransformWrite(const T& value)
        : _dims(_converter.get_size(value))
        , _converter(_dims)
        , _data(value)
    {}

    const dataspace_type* get_pointer() {
        return _converter.get_pointer(_data);
    }

  private:
    std::vector<size_t> _dims;
    Conv _converter;
    const T& _data;
};

template <typename T>
TransformWrite<T> make_transform_write(const T& value) {
    return TransformWrite<T>{value};
}

template <typename T>
TransformRead<T> make_transform_read(const std::vector<size_t>& dims) {
    return TransformRead<T>{dims};
}

}  // namespace HighFive

#ifdef H5_USE_EIGEN
#include "H5ConverterEigen_misc.hpp"
#endif

#endif // H5CONVERTER_MISC_HPP
