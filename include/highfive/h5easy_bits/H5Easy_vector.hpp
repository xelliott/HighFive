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

    inline static DataSet dump(File& file,
                               const std::string& path,
                               const T& data,
                               const DumpOptions& options) {
        using value_type = typename HighFive::details::data_converter<T>::dataspace_type;
        DataSet dataset = initDataset<value_type>(file, path, HighFive::compute_dims(data), options);
        dataset.write(data);
        if (options.flush()) {
            file.flush();
        }
        return dataset;
    }

    inline static T load(const File& file, const std::string& path) {
        return file.getDataSet(path).read<T>();
    }

   inline static Attribute dumpAttribute(File& file,
                                         const std::string& path,
                                         const std::string& key,
                                         const T& data,
                                         const DumpOptions& options) {
        using value_type = typename HighFive::details::data_converter<T>::dataspace_type;
        std::vector<size_t> shape = get_dim_vector(data);
        Attribute attribute = initAttribute<value_type>(file, path, key, shape, options);
        attribute.write(data);
        if (options.flush()) {
            file.flush();
        }
        return attribute;
    }

    inline static T loadAttribute(const File& file,
                                  const std::string& path,
                                  const std::string& key) {
        DataSet dataset = file.getDataSet(path);
        Attribute attribute = dataset.getAttribute(key);
        return attribute.read<T>();
    }
};

}  // namespace detail
}  // namespace H5Easy

#endif  // H5EASY_BITS_VECTOR_HPP
