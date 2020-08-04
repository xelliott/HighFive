/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5EASY_BITS_VECTOR_HPP
#define H5EASY_BITS_VECTOR_HPP

#include "../H5Easy.hpp"
#include "H5Easy_misc.hpp"
#include "H5Easy_scalar.hpp"

namespace H5Easy {

namespace detail {

template <class T>
struct is_vector : std::false_type {};
template <class T>
struct is_vector<std::vector<T>> : std::true_type {};

template <typename T>
struct io_impl<T, typename std::enable_if<is_vector<T>::value>::type> {

    static DataSet dump(File& file, const std::string& path, const T& data) {
        using type_name = typename HighFive::details::data_converter<T>::dataspace_type;
        detail::createGroupsToDataSet(file, path);
        DataSet dataset = file.createDataSet<type_name>(path, DataSpace::From(data));
        dataset.write(data);
        file.flush();
        return dataset;
    }

    static DataSet overwrite(File& file, const std::string& path, const T& data) {
        DataSet dataset = file.getDataSet(path);
        if (HighFive::compute_dims(data) != dataset.getDimensions()) {
            throw detail::error(file, path, "H5Easy::dump: Inconsistent dimensions");
        }
        dataset.write(data);
        file.flush();
        return dataset;
    }

    static T load(const File& file, const std::string& path) {
        DataSet dataset = file.getDataSet(path);
        T data = dataset.read<T>();
        return data;
    }
};

}  // namespace detail
}  // namespace H5Easy

#endif  // H5EASY_BITS_VECTOR_HPP
