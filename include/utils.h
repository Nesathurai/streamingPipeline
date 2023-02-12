


// Hash function for Eigen matrix and vector.
// The code is from `hash_combine` function of the Boost library. See
// http://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine .
template <typename T>
struct matrix_hash : std::unary_function<T, size_t>
{
    std::size_t operator()(T const &matrix) const
    {
        // Note that it is oblivious to the storage order of Eigen matrix (column- or
        // row-major). It will give you the same hash value for two different matrices if they
        // are the transpose of each other in different storage order.
        size_t seed = 0;
        for (size_t i = 0; i < (unsigned long)matrix.size(); ++i)
        {
            auto elem = *(matrix.data() + i);
            seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

bool AreSame(double a, double b)
{
    return fabs(a - b) < 0.1;
}