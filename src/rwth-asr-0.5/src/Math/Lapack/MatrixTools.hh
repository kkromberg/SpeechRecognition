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
#ifndef _MATH_MATRIX_TOOLS_HH_
#define _MATH_MATRIX_TOOLS_HH

#include <Core/Application.hh>
#include <Math/Matrix.hh>
#include <Math/Vector.hh>
#include "Lapack.hh"
#include "Svd.hh"

namespace Math { namespace Lapack {

    template<typename T, class P>  void invert(Math::Matrix<T,P>& mat){
	require(mat.nRows()==mat.nColumns());
	int N = mat.nRows();
	int *ipiv = new int[N];
	T *contMatrix = new T[N*N];
	int info;

	// Pack matrix in Fortran format
	for (int i=0; i<N; i++)
	    for (int j=0; j<N; j++)
		contMatrix[i+j*N] = mat[i][j];

	// LU decomposition
	getrf(&N, &N, contMatrix, &N, ipiv, &info);
	if (info)
	    Core::Application::us()->criticalError("LAPACK error: info=")
		    << info << " in xGETRF";

	// query and set buffer size
	int lwork = -1;
	T *work = new T[1];
	getri(&N, contMatrix, &N, ipiv, work, &lwork, &info);
	if (info)
	    Core::Application::us()->criticalError("LAPACK error: info=")
		    << info << " in xGETRI";
	lwork = int(*work);

	delete[] work;
	work = new T[lwork];

	// actual inverse calculation
	getri(&N, contMatrix, &N, ipiv, work, &lwork, &info);
	if (info)
	    Core::Application::us()->criticalError("LAPACK error: info=")
		    << info << " in xGETRI";

	// unpack into original matrix
	for (int i=0; i<N; i++)
	    for (int j=0; j<N; j++)
		mat[i][j] = contMatrix[i+j*N];

	delete[] ipiv;
	delete[] contMatrix;
	delete[] work;
    }


    template<typename T, class P> void pseudoInvert(Math::Matrix<T,P>& mat){
	// Use existing implementation in Math/Lapack/Svd.cc, but change interface
	DoubleMatrix in(mat);
	DoubleMatrix out;
	pseudoInvert(out, in);
	mat = out;
    }


    template<typename T, class P> T logDeterminant(const Math::Matrix<T,P>& mat){
	require(mat.nRows()==mat.nColumns());
	int N = mat.nRows();
	int *ipiv = new int[N];
	T *contMatrix = new T[N*N];
	int info;

	// Pack matrix in Fortran format
	for (int i=0; i<N; i++)
	    for (int j=0; j<N; j++)
		contMatrix[i+j*N] = mat[i][j];

	// LU decomposition
	getrf(&N, &N, contMatrix, &N, ipiv, &info);
	if (info)
	    Core::Application::us()->criticalError("LAPACK error: info=")
		    << info << " in xGETRF";

	// Sum of diagonal
	T d = 0.0;
	for (u32 i=0; i< mat.nRows(); i++)
	    d += log(fabs(contMatrix[i+i*N]));

	delete[] ipiv;
	delete[] contMatrix;

	return d;
    }

    template<typename T, class P> T determinant(const Math::Matrix<T,P>& mat){
	require(mat.nRows()==mat.nColumns());
	int N = mat.nRows();
	int *ipiv = new int[N];
	T *contMatrix = new T[N*N];
	int info;

	// Pack matrix in Fortran format
	for (int i=0; i<N; i++)
	    for (int j=0; j<N; j++)
		contMatrix[i+j*N] = mat[i][j];

	// LU decomposition
	getrf(&N, &N, contMatrix, &N, ipiv, &info);
	if (info)
	    Core::Application::us()->criticalError("LAPACK error: info=")
		    << info << " in xGETRF";

	// Product of diagonal
	T d = 1.0;
	for (u32 i=0; i< mat.nRows(); i++)
	    d *= fabs(contMatrix[i+i*N]);

	delete[] ipiv;
	delete[] contMatrix;

	return d;
    }
} } // namespace Math::Nr

#endif // _MATH_MATRIX_TOOLS_HH_
