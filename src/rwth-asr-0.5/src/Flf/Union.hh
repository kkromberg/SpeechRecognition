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
#ifndef _FLF_UNION_HH
#define _FLF_UNION_HH

#include <Core/Vector.hh>

#include "FlfCore/Lattice.hh"
#include "Network.hh"


namespace Flf {

    /**
     * union
     **/
    ConstLatticeRef unite(ConstLatticeRef l1, ConstLatticeRef l2,
			  ConstSemiringRef semiring = ConstSemiringRef());
    ConstLatticeRef unite(const ConstLatticeRefList &lats,
			  ConstSemiringRef semiring = ConstSemiringRef());

    NodeRef createUnionNode(const std::string &name, const Core::Configuration &config);

    /**
     * time or boundary conditioned lattice;
     * join either all states having identical boundaries(boundary information includes time) or having identical time
     **/
    typedef enum {
	MeshTypeFullBoundary,
	MeshTypeTimeBoundary
    } MeshType;
    ConstLatticeRef mesh(const ConstLatticeRefList &lats,
			 ConstSemiringRef semiring = ConstSemiringRef(),
			 MeshType meshType = MeshTypeFullBoundary);
    ConstLatticeRef mesh(ConstLatticeRef l1, ConstLatticeRef l2,
			 ConstSemiringRef semiring = ConstSemiringRef(),
			 MeshType meshType = MeshTypeFullBoundary);
    ConstLatticeRef mesh(ConstLatticeRef l, MeshType meshType = MeshTypeFullBoundary);

    NodeRef createMeshNode(const std::string &name, const Core::Configuration &config);

} // namespace Flf

#endif // _FLF_UNION_HH
