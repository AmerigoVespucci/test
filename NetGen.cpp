#include <cstdlib>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <caffe/caffe.hpp>
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include "H5Cpp.h"


#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

typedef unsigned long long u64;

class NGNet {
public:
	NGNet(	const string& model_file,
				const string& trained_file,
				const string& input_layer_name,
				const string& output_layer_name) {
		model_file_ = model_file;
		trained_file_ = trained_file;
		input_layer_name_ = input_layer_name;
		output_layer_name_ = output_layer_name;
	}
	
	void Init();
	Blob<float>* GetVec(bool b_top, int layer_idx, int branch_idx);
	Blob<float>* GetInputVec() {
		return GetVec(true, input_layer_idx_, input_layer_top_idx_);
	}
	Blob<float>* GetOutputVec() {
		return GetVec(true, output_layer_idx_, output_layer_top_idx_);
	}
	void PrepForInput() {
		net_->ForwardFromTo(0, input_layer_idx_);
	}
	float ComputeOutput() {
		return net_->ForwardFromTo(input_layer_idx_+1, output_layer_idx_);
	}
	int input_layer_dim() { return input_layer_dim_; }
	int input_layer_idx() { return input_layer_idx_; }
	
private:
	shared_ptr<Net<float> > net_;
	int input_layer_idx_;
	int input_layer_top_idx_; // currently the index of the array of top_vectors for this net
	int output_layer_idx_;
	int output_layer_top_idx_; // currently the index of the array of top_vectors for this net
	string model_file_;
	string trained_file_;
	string input_layer_name_ ;
	string output_layer_name_;
	int input_layer_dim_;
};

class NetGen {
public:
	NetGen() {bInit_ = false; }

	void PreInit();
	void Init(	vector<NGNet>& nets,
				const string& word_file_name,
				const string& word_vector_file_name);

	bool  Classify();

private:

	std::vector<pair<float, int> > Predict();


	void Preprocess(const cv::Mat& img,
					std::vector<cv::Mat>* input_channels);

private:
	bool bInit_;
	vector<NGNet>* p_nets_;
	vector<vector<float> > data_recs_;
	vector<float> label_recs_;
	vector<string> words_;
	vector<vector<float> > words_vecs_;
	string word_vector_file_name_;
	int words_per_input_;
	
	int GetClosestWordIndex(vector<float>& VecOfWord, int num_input_vals, 
							vector<pair<float, int> >& SortedBest,
							int NumBestKept);

};

void NGNet::Init(	) {


	input_layer_top_idx_ = 0;
	output_layer_top_idx_ = 0;
	
	/* Load the network. */
	net_.reset(new Net<float>(model_file_, TEST));
	NetParameter param;
	CHECK(ReadProtoFromTextFile(model_file_, &param))
		<< "Failed to parse NetParameter file: " << model_file_;
	for (int ip = 0; ip < param.layer_size(); ip++) {
		LayerParameter layer_param = param.layer(ip);
		if (layer_param.has_inner_product_param()) {
			InnerProductParameter* inner_product_param = layer_param.mutable_inner_product_param();
			int num_output = inner_product_param->num_output();
			if (num_output > 0) {
				inner_product_param->set_num_output(num_output * 2); 
			}
		}
	}
//	//param.mutable_state()->set_phase(phase);
	Net<float> * new_net = new Net<float>(param);
	
	net_->CopyTrainedLayersFrom(trained_file_);

	
	int input_layer_idx = -1;
	for (size_t layer_id = 0; layer_id < net_->layer_names().size(); ++layer_id) {
		if (net_->layer_names()[layer_id] == input_layer_name_) {
			input_layer_idx = layer_id;
			break;
		}
	}
	if (input_layer_idx == -1) {
		LOG(FATAL) << "Unknown layer name " << input_layer_name_;			
	}

	input_layer_idx_ = input_layer_idx;
	
	input_layer_top_idx_ = 0;

	Blob<float>* input_layer = net_->top_vecs()[input_layer_idx_][input_layer_top_idx_];
	input_layer_dim_ = input_layer->shape(1);

	int output_layer_idx = -1;
	for (size_t layer_id = 0; layer_id < net_->layer_names().size(); ++layer_id) {
		if (net_->layer_names()[layer_id] == output_layer_name_) {
			output_layer_idx = layer_id;
			break;
		}
	}
	if (output_layer_idx == -1) {
		LOG(FATAL) << "Unknown layer name " << output_layer_name_;			
	}
	output_layer_idx_ = output_layer_idx;
	
	
}

Blob<float>* NGNet::GetVec(bool b_top, int layer_idx, int branch_idx)
{
	if (b_top) {
		return net_->top_vecs()[layer_idx][branch_idx];
	}
	else {
		return net_->bottom_vecs()[layer_idx][branch_idx];
	}
}


void NetGen::PreInit()
{
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif
}

void NetGen::Init(	vector<NGNet>& nets,
						const string& word_file_name,
						const string& word_vector_file_name) {

	word_vector_file_name_ = word_vector_file_name;
	//output_layer_idx_arr_ = vector<int>(5, -1);

	words_per_input_ = 1;
	words_per_input_ = 4;
	
	p_nets_ = &nets;
	
	for (int in = 0; in < nets.size(); in++) {
		NGNet& net = nets[in];
		net.Init();
	}

	
	
	std::ifstream str_words(word_file_name.c_str(), std::ifstream::in);
	if (str_words.is_open() ) {
		string ln;
		//for (int ic = 0; ic < cVocabLimit; ic++) {
		while (str_words.good()) {
			string w;
			getline(str_words, ln, ' ');
			//VecFile >> w;
			w = ln;
			if (w.size() == 0) {
				break;
			}
			words_.push_back(w);
			words_vecs_.push_back(vector<float>());
			vector<float>& curr_vec = words_vecs_.back();
			int num_input_vals = nets[0].input_layer_dim() / words_per_input_;
			for (int iwv = 0; iwv < num_input_vals; iwv++) {
				if (iwv == num_input_vals - 1) {
					getline(str_words, ln);
				}
				else {
					getline(str_words, ln, ' ');
				}
				float wv;
				//wv = stof(ln);
				wv = (float)atof(ln.c_str());
				curr_vec.push_back(wv);
			}

		}
	}
	//Blob<float>*  input_bottom_vec = net_->top_vecs()[input_layer_idx][input_layer_bottom_idx_];
	

	bInit_ = true;

}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

int  NetGen::GetClosestWordIndex(	vector<float>& VecOfWord, int num_input_vals, 
									vector<pair<float, int> >& SortedBest, int NumBestKept)
{
	float MinDiff = num_input_vals * 2.0f;
	int iMinDiff = -1;
	float ThreshDiff = MinDiff;
	for (int iwv =0; iwv < words_vecs_.size(); iwv++ ) {
		float SumDiff = 0.0f;
		for (int iv = 0; iv < num_input_vals; iv++) {
			float Diff = VecOfWord[iv] - words_vecs_[iwv][iv];
			SumDiff += Diff * Diff;
		}
		if (SumDiff < MinDiff) {
			MinDiff = SumDiff;
			iMinDiff = iwv;
		}
		if (SumDiff < ThreshDiff) {
			SortedBest.push_back(make_pair(SumDiff, iwv));
			std::sort(SortedBest.begin(), SortedBest.end());
			if (SortedBest.size() > NumBestKept) {
				SortedBest.pop_back();
				ThreshDiff = SortedBest.back().first;
			}
		}
	}
	return iMinDiff;
}

/* Return the values in the output layer */
bool NetGen::Classify() {
	CHECK(bInit_) << "NetGen: Init must be called first\n";
	
	vector<pair<string, vector<float> > > VecArr;
	int num_vals_per_word = (*p_nets_)[0].input_layer_dim() / words_per_input_;
	Blob<float>* predict_input_layer = (*p_nets_)[0].GetInputVec();
	Blob<float>* predict_label_layer = (*p_nets_)[0].GetVec(
		true, (*p_nets_)[0].input_layer_idx(), 1);
	Blob<float>* valid_input_layer = (*p_nets_)[1].GetInputVec();
	Blob<float>* predict_output_layer = (*p_nets_)[0].GetOutputVec();
	Blob<float>* valid_output_layer = (*p_nets_)[1].GetOutputVec();
	
	int CountMatch = 0;
	int NumTestRecs = 5000;
	for (int ir = 0; ir < NumTestRecs; ir++) {
		(*p_nets_)[0].PrepForInput();
		const float* p_in = predict_input_layer->cpu_data();  // ->cpu_data();
		const float* p_lbl = predict_label_layer->cpu_data();
		//net_->ForwardFromTo(0, input_layer_idx_);

		//string w = words_[isym];
		//std::cerr << w << ",";
		const int cNumInputWords = 4;
		const int cNumValsPerWord = 100;
		int iMinDiffLbl;
		vector<pair<float, int> > SortedBest;
		vector<pair<float, int> > SortedBestDummy;
		vector<int> ngram_indices(5, -1);
		int cNumBestKept = 10;
		for (int iw = 0; iw < cNumInputWords; iw++) {
			vector<float> VecOfWord;
			vector<float> VecOfLabel;
			for (int iact = 0; iact < cNumValsPerWord; iact++) {
				//*p_in++ = words_vecs_[isym][iact];
				VecOfWord.push_back(*p_in++);
			}
			if (iw == 1) {				
				for (int iact = 0; iact < cNumValsPerWord; iact++) {
					//*p_in++ = words_vecs_[isym][iact];
					VecOfLabel.push_back(*p_lbl++);
				}
				iMinDiffLbl = GetClosestWordIndex(VecOfLabel, num_vals_per_word,
													SortedBestDummy, 1);
			}
			int iMinDiff = GetClosestWordIndex(VecOfWord, num_vals_per_word, SortedBestDummy, 1);
			if (iMinDiff != -1) {
				int word_idx = ((iw <= 1) ? iw : iw+1);
				ngram_indices[word_idx] = iMinDiff;
				string w = words_[iMinDiff];
				std::cerr << w << " ";
				if (iw == 1) {
					if (iMinDiffLbl == -1) {
						std::cerr << "XXX ";
					}
					else {
						string l = words_[iMinDiffLbl];
						std::cerr << "(" << l << ") ";
					}
				}
			}
		}
		std::cerr << std::endl;

		float loss = (*p_nets_)[0].ComputeOutput();
		//float loss = net_->ForwardFromTo(input_layer_idx_+1, output_layer_idx_);

		const float* p_out = predict_output_layer->cpu_data();  
		vector<float> output;
		vector<float> VecOfWord;
		for (int io = 0; io < predict_output_layer->shape(1); io++) {
			float data = *p_out++;
			VecOfWord.push_back(data);
		}
		int iMinDiff = GetClosestWordIndex(	VecOfWord, num_vals_per_word,
											SortedBest, cNumBestKept);
		if (iMinDiff != -1) {
			std::cerr << "--> ";
			vector <pair<float, int> > ReOrdered;
			for (int ib = 0; ib < SortedBest.size(); ib++) {
				ngram_indices[2] = SortedBest[ib].second;
				(*p_nets_)[1].PrepForInput();
				float* p_v_in = valid_input_layer->mutable_cpu_data();  // ->cpu_data();
				const int cNumValidInputWords = 5;
				for (int iw = 0; iw < cNumValidInputWords; iw++) {
					for (int iact = 0; iact < cNumValsPerWord; iact++) {
						*p_v_in++ = words_vecs_[ngram_indices[iw]][iact];
					}
				}
				float v_loss = (*p_nets_)[1].ComputeOutput();
				const float* p_v_out = valid_output_layer->cpu_data();  
				float v_val = p_v_out[1]; // seems p_v_out[0] is 1 - p_v_out[1]
				
				string w = words_[SortedBest[ib].second];
				ReOrdered.push_back(make_pair(SortedBest[ib].first / (v_val * v_val * v_val), SortedBest[ib].second));
				std::cerr << w << " (" << SortedBest[ib].first << " vs. " << v_val << "), ";
			}
			std::cerr << std::endl << "Reordered: ";
			std::sort(ReOrdered.begin(), ReOrdered.end());
			for (int iro = 0; iro < ReOrdered.size(); iro++) {
				std::cerr <<  words_[ReOrdered[iro].second] << " (" << ReOrdered[iro].first << "), ";
			}
			
			if (iMinDiffLbl == ReOrdered.front().second) {
				CountMatch++;
			}
		}
		std::cerr << std::endl;
		//VecArr.push_back(make_pair(w, output));
	}
	
	std::cerr << CountMatch << " records hit exactly out of " << NumTestRecs << "\n";
	
	std::ofstream str_vecs(word_vector_file_name_.c_str());
	if (str_vecs.is_open()) { 
		//str_vecs << VecArr[0].second.size() << " ";
		for (int iv = 0; iv < VecArr.size(); iv++) {
			pair<string, vector<float> >& rec = VecArr[iv];
			str_vecs << rec.first << " ";
			vector<float>& vals = rec.second;
			for (int ir = 0; ir < vals.size(); ir++) {
				str_vecs << vals[ir];
				if (ir == vals.size() - 1) {
					str_vecs << std::endl;
				}
				else {
					str_vecs << " ";
				}
			}
		}
	}



	return true;
}


/*
 /home/abba/caffe/toys/ValidClicks/train.prototxt /guten/data/ValidClicks/data/v.caffemodel
 /home/abba/caffe/toys/SimpleMoves/Forward/train.prototxt /devlink/caffe/data/SimpleMoves/Forward/models
 */

#ifdef CAFFE_NET_GEN_MAIN
int main(int argc, char** argv) {
//	if (argc != 3) {
//		std::cerr << "Usage: " << argv[0]
//				  << " deploy.prototxt network.caffemodel" << std::endl;
//		return 1;
//	 }

	FLAGS_log_dir = "/devlink/caffe/log";
	::google::InitGoogleLogging(argv[0]);
  
//	string model_file   = "/home/abba/caffe-recurrent/toys/NetGen/VecPredict/train.prototxt";
//	string trained_file = "/devlink/caffe/data/NetGen/VecPredict/models/v_iter_500000.caffemodel";
	string word_file_name = "/devlink/caffe/data/WordEmbed/VecPredict/data/WordList.txt";
	string word_vector_file_name = "/devlink/caffe/data/WordEmbed/VecPredict/data/WordVectors.txt";
//	string model_file   = "/home/abba/caffe-recurrent/toys/LSTMTrain/WordToPos/train.prototxt";
//	string trained_file = "/devlink/caffe/data/LSTMTrain/WordToPos/models/a_iter_1000000.caffemodel";
//	string word_file_name = "/devlink/caffe/data/LSTMTrain/WordToPos/data/WordList.txt";
//	string word_vector_file_name = "/devlink/caffe/data/LSTMTrain/WordToPos/data/WordVectors.txt";
	string input_layer_name = "data";
	string output_layer_name = "SquashOutput";
	
	int input_data_idx = 0;
	int input_label_idx = 1;

	NetGen classifier;
	classifier.PreInit();
	vector<NGNet> nets;
	nets.push_back(NGNet(
		"/devlink/github/test/toys/NetGen/VecPredict/train.prototxt",
		"/devlink/caffe/data/NetGen/VecPredict/models/n_iter_82634.caffemodel",
		"data", "SquashOutput"));
//	nets.push_back(NGNet(
//		"/home/abba/caffe-recurrent/toys/WordEmbed/GramValid/train.prototxt",
//		"/devlink/caffe/data/WordEmbed/GramValid/models/g_iter_500000.caffemodel",
//		"data", "SquashOutput"));
	classifier.Init(nets,
					word_file_name, 
					word_vector_file_name);
	classifier.Classify();
	
}
#endif // CAFFE_MULTINET_MAIN