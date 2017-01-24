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
// $Id: Regression.hh 5439 2005-11-09 11:05:06Z bisani $

#ifndef _SIGNAL_REGRESSION_HH
#define _SIGNAL_REGRESSION_HH

#include <Flow/Merger.hh>
#include <Flow/Vector.hh>


namespace Signal {

    /**
     * Dumb port of the old Regression_Module.c
     */

    class Regression {
    protected:
	typedef std::vector<f32> Frame;

    public:
	Regression();
	~Regression();
	void regressFirstOrder (const std::vector<const Frame*> &in, Frame &out);
	void regressSecondOrder(const std::vector<const Frame*> &in, Frame &out);
    };

    class RegressionNode :
	public Flow::MergerNode< Flow::Vector<f32>, Flow::Vector<f32> >,
	private Regression
    {
	typedef Flow::MergerNode< Flow::Vector<f32>, Flow::Vector<f32> > Precursor;
    private:
	u32 order_;
    public:
	static std::string filterName() { return "signal-regression"; }
	static const Core::ParameterInt parameterOrder;

	RegressionNode(const Core::Configuration &c);
	virtual ~RegressionNode();

	virtual bool setParameter(const std::string &name, const std::string &value);
	virtual Precursor::OutputData *merge(std::vector<Precursor::InputFrame>&);
    };

}

#endif // _SIGNAL_REGRESSION_HH
