// Copyright 2011 RWTH Aachen University. All rights reserved.
//
// Licensed under the RWTH ASR License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef _MATH_UTILITIES_HH
#define _MATH_UTILITIES_HH

#include <numeric>

namespace Math {

    /** absolute difference function
     * @return |x - y|
     */
    template<class T>
    struct absoluteDifference : public std::binary_function<T, T, T> {
	T operator()(T x, T y) { return Core::abs(x - y); }
    };


    /** absolute difference to the power function
     * @return |x - y|^power
     */
    template<class T>
    class absoluteDifferencePower : public std::binary_function<T, T, T> {
    private:
	f64 power_;
    public:
	absoluteDifferencePower(const f64 power) : power_(power) {}

	T operator()(T x, T y) { return (T)pow(Core::abs(x - y), power_); }
    };

    /** absolute difference square-root function
     * @return |x - y|^0.5
     */
    template<class T>
    struct absoluteDifferenceSquareRoot : public std::binary_function<T, T, T> {
	T operator()(T x, T y) { return (T)sqrt(Core::abs(x - y)); }
    };

    /** absolute difference square function
     * @return |x - y|^2
     */
    template<class T>
    struct absoluteDifferenceSquare : public std::binary_function<T, T, T> {
	T operator()(T x, T y) { return (x - y) * (x - y); }
    };

} // namespace Math

#endif // _MATH_UTILITIES_HH

