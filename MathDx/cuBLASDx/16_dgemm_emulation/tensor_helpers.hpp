#pragma once

// For CuTe Tensor types
#include <cublasdx.hpp>

// For required cuda::std types
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/std/type_traits>

namespace example {

    namespace detail {

        template<class Element>
        CUBLASDX_HOST_DEVICE auto convert_to_cute_tuple_element(Element const& elem) {
            static_assert(cuda::std::is_integral_v<Element>, "Only flat integral tuples are supported");
            return elem;
        }

        template<class Element, Element Value>
        CUBLASDX_HOST_DEVICE auto convert_to_cute_tuple_element(cuda::std::integral_constant<Element, Value>) {
            static_assert(cuda::std::is_integral_v<Element>, "Only flat integral tuples are supported");
            return cute::Int<Value> {};
        }

        template<class... TupleArgs, int... Indices>
        CUBLASDX_HOST_DEVICE auto convert_to_cute_tuple(cuda::std::tuple<TupleArgs...> const& std_tuple,
                                                        cuda::std::integer_sequence<int, Indices...>) {
            return cute::make_tuple(convert_to_cute_tuple_element(cuda::std::get<Indices>(std_tuple))...);
        }

        template<class... TupleArgs>
        CUBLASDX_HOST_DEVICE auto convert_to_cute_tuple(cuda::std::tuple<TupleArgs...> const& std_tuple) {
            constexpr unsigned num_elems = sizeof...(TupleArgs);
            return convert_to_cute_tuple(std_tuple, cuda::std::make_integer_sequence<int, num_elems>());
        }
    } // namespace detail

    template<class PointerType, class... ShapeArgs, class... StrideArgs>
    CUBLASDX_HOST_DEVICE auto make_gmem_tensor_from_tuples(PointerType*                           pointer_type,
                                                           cuda::std::tuple<ShapeArgs...> const&  shape,
                                                           cuda::std::tuple<StrideArgs...> const& stride) {

        auto cute_shape  = detail::convert_to_cute_tuple(shape);
        auto cute_stride = detail::convert_to_cute_tuple(stride);
        auto cute_layout = cute::make_layout(cute_shape, cute_stride);

        return cute::make_tensor(cute::make_gmem_ptr(pointer_type), cute_layout);
    }

    template<class... ShapeArgs, class... StrideArgs>
    CUBLASDX_HOST_DEVICE auto make_layout_from_tuples(cuda::std::tuple<ShapeArgs...> const&  shape,
                                                      cuda::std::tuple<StrideArgs...> const& stride) {

        auto cute_shape  = detail::convert_to_cute_tuple(shape);
        auto cute_stride = detail::convert_to_cute_tuple(stride);
        return cute::make_layout(cute_shape, cute_stride);
    }

    template<class Tensor>
    struct tensor_value_type;

    template<class Engine, class Layout>
    struct tensor_value_type<cublasdx::tensor<Engine, Layout>> {
        using type = typename Engine::value_type;
    };

    template<class T>
    using tensor_value_type_t = typename tensor_value_type<T>::type;

    using cute::conditional_return;
} // namespace example
